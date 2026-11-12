"""Independent RIFF/WAVE reference parser used as the generative-test oracle.

This parser is written from scratch against the public RIFF/WAVE container
rules and the service's own published media contract (16-bit integer PCM,
mono or stereo, 8000-192000 Hz). It deliberately shares no code with the
production Go decoder in internal/audiocodec:

* chunk discovery is bounded by the RIFF size declared in the header;
* every data-bearing chunk inside the RIFF body contributes to the logical
  payload, instead of a later chunk silently replacing an earlier one;
* legal but uncommon container organisations (unknown chunks, LIST chunks,
  odd-sized chunks with the standard pad byte, fmt/data in either order,
  several data chunks) are preserved and decoded;
* only genuinely invalid media is rejected, with a structured reason.

The module has no third-party dependencies and never imports the project
tree, so an injected module next to the service cannot influence verdicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

RIFF_TAG = b"RIFF"
WAVE_TAG = b"WAVE"
FMT_TAG = b"fmt "
DATA_TAG = b"data"
LIST_TAG = b"LIST"
JUNK_TAGS = (b"JUNK", b"junk", b"PAD ", b"bext", b"fact")

PCM_FORMAT_CODE = 1
PCM_BITS = 16
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
SILENCE_FLOOR_DBFS = -120.0


class OracleReject(Exception):
    """The oracle considers the bytes invalid media for this service."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Chunk:
    identifier: bytes
    start: int  # absolute offset of the chunk identifier
    size: int
    payload_start: int
    payload_end: int
    pad_byte: bool  # whether a trailing alignment pad byte is required


@dataclass
class FormatInfo:
    audio_format: int
    channels: int
    sample_rate: int
    byte_rate: int
    block_align: int
    bits_per_sample: int
    extension_size: Optional[int] = None
    valid_bits_per_sample: Optional[int] = None
    channel_mask: Optional[int] = None
    sub_format: Optional[bytes] = None


@dataclass
class WaveVerdict:
    """What an independent observer concludes the media actually is."""

    accepted: bool
    reject_reason: Optional[str] = None
    reject_detail: str = ""
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bits_per_sample: Optional[int] = None
    frame_count: Optional[int] = None
    data_bytes: int = 0
    chunk_count: int = 0
    data_chunk_count: int = 0
    fmt_chunk_count: int = 0
    unknown_chunk_count: int = 0
    samples: List[float] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    # Frames contributed by each data chunk in document order.
    data_frame_parts: List[int] = field(default_factory=list)

    def reject(self, reason: str, detail: str = "") -> "WaveVerdict":
        self.accepted = False
        self.reject_reason = reason
        self.reject_detail = detail
        return self


def _u16(buffer: bytes, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 2], "little", signed=False)


def _u32(buffer: bytes, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 4], "little", signed=False)


def _read_chunks(data: bytes, body_end: int) -> List[Chunk]:
    """Walk RIFF chunks, stopping at the declared RIFF boundary.

    The RIFF size field owns the container extent: bytes after
    ``body_end`` are trailing data, not chunks. A chunk whose declared
    extent crosses that boundary is malformed.
    """
    chunks: List[Chunk] = []
    offset = 12
    while offset < body_end:
        remaining = body_end - offset
        if remaining < 8:
            # A partial chunk header inside the RIFF body is malformed.
            raise OracleReject("truncated_chunk_header", f"only {remaining} bytes remain at offset {offset}")
        identifier = data[offset : offset + 4]
        size = _u32(data, offset + 4)
        payload_start = offset + 8
        payload_end = payload_start + size
        if payload_end > body_end:
            raise OracleReject(
                "chunk_exceeds_riff_body",
                f"chunk {identifier!r} at {offset} declares {size} bytes, body ends at {body_end}",
            )
        pad = size % 2 == 1
        chunks.append(
            Chunk(
                identifier=identifier,
                start=offset,
                size=size,
                payload_start=payload_start,
                payload_end=payload_end,
                pad_byte=pad,
            )
        )
        offset = payload_end + (1 if pad else 0)
    return chunks


def _parse_format(payload: bytes) -> FormatInfo:
    if len(payload) < 16:
        raise OracleReject("fmt_too_short", f"fmt payload is {len(payload)} bytes")
    info = FormatInfo(
        audio_format=_u16(payload, 0),
        channels=_u16(payload, 2),
        sample_rate=_u32(payload, 4),
        byte_rate=_u32(payload, 8),
        block_align=_u16(payload, 12),
        bits_per_sample=_u16(payload, 14),
    )
    if len(payload) >= 18:
        info.extension_size = _u16(payload, 16)
    # A WAVEFORMATEXTENSIBLE header carries 22 extension bytes after the
    # 16-byte base plus the 2-byte extension length.
    if info.audio_format == 0xFFFE and len(payload) >= 40:
        info.valid_bits_per_sample = _u16(payload, 18)
        info.channel_mask = _u32(payload, 20)
        info.sub_format = bytes(payload[24:40])
    return info


def _decode_pcm16(payload: bytes) -> List[float]:
    samples: List[float] = []
    for index in range(0, len(payload) - 1, 2):
        raw = int.from_bytes(payload[index : index + 2], "little", signed=True)
        if raw < 0:
            samples.append(raw / 32768.0)
        else:
            samples.append(raw / 32767.0)
    return samples


def parse_wav(data: bytes) -> WaveVerdict:
    """Return an independent verdict for candidate WAV bytes."""
    verdict = WaveVerdict(accepted=False)
    if len(data) < 12:
        return verdict.reject("shorter_than_riff_header", f"{len(data)} bytes")
    if data[0:4] != RIFF_TAG or data[8:12] != WAVE_TAG:
        return verdict.reject("not_riff_wave", "missing RIFF/WAVE signature")

    riff_size = _u32(data, 4)
    body_end = riff_size + 8
    if body_end > len(data):
        return verdict.reject(
            "riff_size_exceeds_payload",
            f"declared {body_end} bytes, payload is {len(data)}",
        )

    try:
        chunks = _read_chunks(data, body_end)
    except OracleReject as reject:
        return verdict.reject(reject.reason, reject.detail)
    verdict.chunks = chunks
    verdict.chunk_count = len(chunks)

    fmt_chunks = [chunk for chunk in chunks if chunk.identifier == FMT_TAG]
    data_chunks = [chunk for chunk in chunks if chunk.identifier == DATA_TAG]
    verdict.fmt_chunk_count = len(fmt_chunks)
    verdict.data_chunk_count = len(data_chunks)
    verdict.unknown_chunk_count = sum(
        1 for chunk in chunks if chunk.identifier not in (FMT_TAG, DATA_TAG)
    )

    if not fmt_chunks:
        return verdict.reject("missing_fmt_chunk", "no fmt chunk inside the RIFF body")
    if not data_chunks:
        return verdict.reject("missing_data_chunk", "no data chunk inside the RIFF body")
    if len(fmt_chunks) > 1:
        # Two conflicting format descriptions make the media ambiguous;
        # this is structural invalidity rather than an uncommon layout.
        return verdict.reject("duplicate_fmt_chunk", f"{len(fmt_chunks)} fmt chunks")

    try:
        fmt_info = _parse_format(bytes(data[fmt_chunks[0].payload_start : fmt_chunks[0].payload_end]))
    except OracleReject as reject:
        return verdict.reject(reject.reason, reject.detail)

    if fmt_info.audio_format == 0xFFFE and fmt_info.sub_format is not None:
        # Extensible PCM: the first two bytes of the KSDATAFORMAT_SUBTYPE
        # GUID equal the legacy format tag, which must be PCM.
        if _u16(fmt_info.sub_format, 0) != PCM_FORMAT_CODE:
            return verdict.reject("unsupported_format_code", "extensible sub-format is not PCM")
        if fmt_info.valid_bits_per_sample not in (None, PCM_BITS):
            return verdict.reject(
                "unsupported_pcm_width",
                f"extensible valid bits {fmt_info.valid_bits_per_sample}",
            )
    elif fmt_info.audio_format != PCM_FORMAT_CODE:
        return verdict.reject("unsupported_format_code", f"format tag {fmt_info.audio_format}")
    if fmt_info.bits_per_sample != PCM_BITS:
        return verdict.reject("unsupported_pcm_width", f"{fmt_info.bits_per_sample} bits")
    if fmt_info.channels not in (1, 2):
        return verdict.reject("unsupported_channels", f"{fmt_info.channels} channels")
    if not MIN_SAMPLE_RATE <= fmt_info.sample_rate <= MAX_SAMPLE_RATE:
        return verdict.reject("sample_rate_out_of_range", str(fmt_info.sample_rate))

    expected_block_align = fmt_info.channels * (PCM_BITS // 8)
    if fmt_info.block_align != expected_block_align:
        return verdict.reject(
            "block_align_mismatch",
            f"declared {fmt_info.block_align}, expected {expected_block_align}",
        )
    expected_byte_rate = fmt_info.sample_rate * expected_block_align
    if fmt_info.byte_rate != expected_byte_rate:
        return verdict.reject(
            "byte_rate_mismatch",
            f"declared {fmt_info.byte_rate}, expected {expected_byte_rate}",
        )

    logical_payload = bytearray()
    for chunk in data_chunks:
        part = data[chunk.payload_start : chunk.payload_end]
        if len(part) == 0:
            return verdict.reject("empty_data_chunk", f"data chunk at offset {chunk.start}")
        if len(part) % expected_block_align != 0:
            return verdict.reject(
                "data_not_frame_aligned",
                f"data chunk at offset {chunk.start} is {len(part)} bytes",
            )
        logical_payload.extend(part)

    if not logical_payload:
        return verdict.reject("empty_payload", "all data chunks are empty")

    samples = _decode_pcm16(bytes(logical_payload))
    frame_count = len(samples) // fmt_info.channels
    if frame_count < 1:
        return verdict.reject("empty_source", "decoded payload has no full frame")

    verdict.accepted = True
    verdict.sample_rate = fmt_info.sample_rate
    verdict.channels = fmt_info.channels
    verdict.bits_per_sample = PCM_BITS
    verdict.frame_count = frame_count
    verdict.data_bytes = len(logical_payload)
    verdict.samples = samples
    verdict.data_frame_parts = [
        (chunk.payload_end - chunk.payload_start) // expected_block_align for chunk in data_chunks
    ]
    return verdict


def measure(samples: List[float]) -> Tuple[float, float]:
    """Peak and RMS in dBFS, computed independently of the production code."""
    if not samples:
        return SILENCE_FLOOR_DBFS, SILENCE_FLOOR_DBFS
    peak = 0.0
    sum_squares = 0.0
    for sample in samples:
        absolute = abs(sample)
        if absolute > peak:
            peak = absolute
        sum_squares += sample * sample
    rms = math.sqrt(sum_squares / len(samples))
    return amplitude_dbfs(peak), amplitude_dbfs(rms)


def amplitude_dbfs(value: float) -> float:
    if value <= 0.0:
        return SILENCE_FLOOR_DBFS
    return max(20.0 * math.log10(value), SILENCE_FLOOR_DBFS)
