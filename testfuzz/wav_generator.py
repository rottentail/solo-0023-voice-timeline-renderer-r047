"""Deterministic WAV container generator and structural mutators.

The generator separates two concerns:

* :func:`build_wav` assembles a RIFF/WAVE container from an explicit chunk
  plan, so legal-but-uncommon organisations (extra chunks, LIST wrappers,
  pad bytes, reordered or split data) can be expressed without hand-rolling
  bytes at every call site;
* :func:`generate_corpus` produces a fixed, seeded corpus in which every
  case carries a record of whether an independent judge must accept it.

All mutators operate on the container while keeping the original PCM bytes
reachable, which is what lets the reducer preserve the original data error
instead of shrinking down to a bare crash input.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .oracle import wav_oracle

# A chunk plan item is (fourcc, payload, odd_size). When odd_size is true the
# chunk is emitted with an odd declared size plus the standard zero alignment
# byte, exercising the RIFF padding path.
ChunkPlan = Tuple[bytes, bytes, bool]


def _chunk(fourcc: bytes, payload: bytes, odd_size: bool = False) -> ChunkPlan:
    return (fourcc, payload, odd_size)


def _fmt_payload(
    sample_rate: int,
    channels: int,
    *,
    audio_format: int = wav_oracle.PCM_FORMAT_CODE,
    bits: int = 16,
    block_align_override: Optional[int] = None,
    byte_rate_override: Optional[int] = None,
    extension: Optional[bytes] = None,
) -> bytes:
    block_align = block_align_override if block_align_override is not None else channels * (bits // 8)
    byte_rate = byte_rate_override if byte_rate_override is not None else sample_rate * block_align
    payload = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
    )
    if extension is not None:
        payload += struct.pack("<H", len(extension)) + extension
    return bytes(payload)


def _pcm_bytes(samples: List[float]) -> bytes:
    out = bytearray()
    for sample in samples:
        if sample <= -1.0:
            quantized = -32768
        elif sample >= 1.0:
            quantized = 32767
        elif sample < 0:
            quantized = round(sample * 32768.0)
        else:
            quantized = round(sample * 32767.0)
        out += struct.pack("<h", quantized)
    return bytes(out)


def sine_samples(
    sample_rate: int,
    channels: int,
    frames: int,
    frequency: float,
    amplitude: float,
    phase: float = 0.0,
) -> List[float]:
    samples: List[float] = []
    for frame in range(frames):
        seconds = frame / sample_rate
        first = math.sin(2 * math.pi * frequency * seconds + phase) * amplitude
        if channels == 1:
            samples.append(first)
            continue
        second = math.sin(2 * math.pi * frequency * seconds + phase + 0.35) * amplitude * 0.8
        samples.extend((first, second))
    return samples


def build_wav(plan: List[ChunkPlan], *, riff_size_override: Optional[int] = None) -> bytes:
    body = bytearray()
    for fourcc, payload, odd_size in plan:
        declared_size = len(payload)
        body += fourcc
        if odd_size and declared_size % 2 == 0:
            declared_size += 1
        body += struct.pack("<I", declared_size)
        body += payload
        if declared_size % 2 == 1:
            body += b"\x00"
    riff_size = riff_size_override if riff_size_override is not None else len(body) + 4
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + bytes(body)


def standard_wav(sample_rate: int, channels: int, samples: List[float]) -> bytes:
    pcm = _pcm_bytes(samples)
    return build_wav(
        [
            _chunk(b"fmt ", _fmt_payload(sample_rate, channels)),
            _chunk(b"data", pcm),
        ]
    )


@dataclass
class CorpusCase:
    name: str
    description: str
    # expectation is True when the independent oracle and the service must
    # both accept; False when both must reject.
    expectation: bool
    # semantic marks cases whose divergence destroys media facts rather than
    # merely changing accept/reject status; the reducer protects the facts.
    semantic: bool
    builder: Callable[[], bytes]


def _extensible_pcm_fmt(sample_rate: int, channels: int) -> bytes:
    block_align = channels * 2
    # WAVEFORMATEXTENSIBLE: 16-byte base, cbSize=22, then wValidBitsPerSample
    # (2), dwChannelMask (4), and KSDATAFORMAT_SUBTYPE_PCM (16-byte GUID
    # whose Data1 field equals the legacy PCM format tag).
    base = struct.pack(
        "<HHIIHH",
        0xFFFE,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        16,
    )
    extension = (
        struct.pack("<H", 16)  # wValidBitsPerSample
        + struct.pack("<I", 0)  # dwChannelMask
        + struct.pack("<I", 1)  # KSDATAFORMAT_SUBTYPE_PCM GUID, Data1 = PCM tag
        + b"\x00" * 12  # Data2, Data3, Data4: GUID is 16 bytes total
    )
    assert len(extension) == 22
    return base + struct.pack("<H", len(extension)) + extension


def generate_corpus() -> List[CorpusCase]:
    """Return the fixed generative corpus.

    The corpus intentionally mixes:
    * canonical encodings;
    * legal-but-uncommon containers that a permissive parser must accept;
    * structurally invalid media that every parser must reject;
    * containers that stay WAV-shaped but lie about their own payload,
      which is the class the old smoke test could never observe.
    """
    rate = 16000
    channels = 1
    frames = 48  # three milliseconds: tiny, cheap, and still multi-frame
    samples = sine_samples(rate, channels, frames, 440.0, 0.4)
    pcm = _pcm_bytes(samples)

    stereo_rate = 24000
    stereo_samples = sine_samples(stereo_rate, 2, frames, 330.0, 0.3)

    cases: List[CorpusCase] = []

    def add(name: str, description: str, expectation: bool, semantic: bool, builder: Callable[[], bytes]) -> None:
        cases.append(CorpusCase(name, description, expectation, semantic, builder))

    # --- canonical media ---------------------------------------------------
    add("canonical_mono", "ordinary mono PCM WAV", True, False, lambda: standard_wav(rate, 1, samples))
    add(
        "canonical_stereo",
        "ordinary stereo PCM WAV",
        True,
        False,
        lambda: standard_wav(stereo_rate, 2, stereo_samples),
    )
    add(
        "minimum_one_frame",
        "smallest possible non-empty mono source",
        True,
        False,
        lambda: standard_wav(rate, 1, [0.25]),
    )

    # --- legal but uncommon container organisations -----------------------
    add(
        "junk_chunk_between",
        "JUNK padding chunk between fmt and data",
        True,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"JUNK", b"\xAB" * 12),
                _chunk(b"data", pcm),
            ]
        ),
    )
    add(
        "list_info_chunk",
        "LIST/INFO metadata chunk before fmt",
        True,
        False,
        lambda: build_wav(
            [
                _chunk(b"LIST", b"INFO" + b"INAM" + struct.pack("<I", 5) + b"take\x00"),
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"data", pcm),
            ]
        ),
    )
    add(
        "odd_sized_unknown_chunk",
        "odd-length unknown chunk exercising RIFF pad bytes",
        True,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"xtra", b"\x01\x02\x03", odd_size=True),
                _chunk(b"data", pcm),
            ]
        ),
    )
    add(
        "data_before_fmt",
        "data chunk preceding the fmt chunk",
        True,
        False,
        lambda: build_wav(
            [
                _chunk(b"data", pcm),
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
            ]
        ),
    )
    add(
        "fmt_with_empty_extension",
        "fmt chunk carrying a zero-length extension field",
        True,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels, extension=b"")),
                _chunk(b"data", pcm),
            ]
        ),
    )
    add(
        "trailing_unknown_after_data",
        "unknown chunk placed after the data chunk",
        True,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"data", pcm),
                _chunk(b"PAD ", b"\x00" * 6),
            ]
        ),
    )
    add(
        "odd_sized_data_chunk",
        "odd declared data size cannot be frame-aligned PCM16 and must be rejected",
        False,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"data", pcm, odd_size=True),
            ]
        ),
    )
    add(
        "extensible_pcm",
        "WAVEFORMATEXTENSIBLE header describing plain PCM",
        True,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _extensible_pcm_fmt(rate, channels)),
                _chunk(b"data", pcm),
            ]
        ),
    )

    # --- split data: legal containers whose facts a replacing parser loses -
    split_point = (len(pcm) // 4) * 2  # frame-aligned split
    first_part, second_part = pcm[:split_point], pcm[split_point:]
    add(
        "two_data_chunks",
        "PCM payload split across two data chunks",
        True,
        True,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"data", first_part),
                _chunk(b"data", second_part),
            ]
        ),
    )
    third = max(2, ((len(pcm) // 3) // 2) * 2)
    add(
        "three_data_chunks",
        "PCM payload split across three data chunks with JUNK between parts",
        True,
        True,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"data", pcm[:third]),
                _chunk(b"JUNK", b"\xCD" * 4),
                _chunk(b"data", pcm[third : third * 2]),
                _chunk(b"data", pcm[third * 2 :]),
            ]
        ),
    )
    add(
        "data_split_by_fmt_order",
        "data, fmt, then a second data chunk continuing the same stream",
        True,
        True,
        lambda: build_wav(
            [
                _chunk(b"data", first_part),
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"data", second_part),
            ]
        ),
    )

    # --- structural invalidity that both decoders must reject --------------
    add("not_wave_plain_bytes", "non-audio byte payload", False, False, lambda: b"not a wave payload at all")
    add(
        "riff_wrong_form_type",
        "RIFF container whose form type is not WAVE",
        False,
        False,
        lambda: b"RIFF" + struct.pack("<I", 4) + b"AVI ",
    )
    add(
        "riff_size_beyond_payload",
        "RIFF size claims bytes that are not present",
        False,
        False,
        lambda: b"RIFF"
        + struct.pack("<I", 5000)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + _fmt_payload(rate, channels),
    )
    add("missing_fmt", "data chunk without any fmt chunk", False, False, lambda: build_wav([_chunk(b"data", pcm)]))
    add(
        "missing_data",
        "fmt chunk without any data chunk",
        False,
        False,
        lambda: build_wav([_chunk(b"fmt ", _fmt_payload(rate, channels))]),
    )
    add(
        "compressed_format",
        "fmt declares a non-PCM format tag",
        False,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels, audio_format=3)),
                _chunk(b"data", pcm),
            ]
        ),
    )
    add(
        "eight_bit_pcm",
        "fmt declares 8-bit PCM outside the service contract",
        False,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels, bits=8, block_align_override=channels)),
                _chunk(b"data", b"\x80" * frames * channels),
            ]
        ),
    )
    add(
        "three_channels",
        "fmt declares an unsupported channel count",
        False,
        False,
        lambda: build_wav(
            [
                _chunk(
                    b"fmt ",
                    _fmt_payload(rate, 6, block_align_override=12, byte_rate_override=rate * 12),
                ),
                _chunk(b"data", pcm[:24]),
            ]
        ),
    )
    add(
        "block_align_lie",
        "fmt block alignment contradicts channels and sample width",
        False,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels, block_align_override=4)),
                _chunk(b"data", pcm),
            ]
        ),
    )
    add(
        "data_not_aligned",
        "data payload length is not frame aligned",
        False,
        False,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"data", pcm[:-1]),
            ]
        ),
    )
    add(
        "chunk_crosses_riff_body",
        "data chunk claims more bytes than the RIFF size allows",
        False,
        False,
        lambda: _chunk_crosses_body(rate, channels, pcm),
    )

    # --- WAV-shaped containers that lie about their own payload ------------
    add(
        "duplicate_fmt_last_wins",
        "two fmt chunks with different sample rates",
        False,
        True,
        lambda: build_wav(
            [
                _chunk(b"fmt ", _fmt_payload(rate, channels)),
                _chunk(b"fmt ", _fmt_payload(8000, channels)),
                _chunk(b"data", pcm),
            ]
        ),
    )
    add(
        "hidden_tail_behind_riff_size",
        "extra data chunk exists past the declared RIFF boundary",
        True,
        True,
        lambda: _hidden_tail(rate, channels, first_part, second_part),
    )
    add(
        "riff_size_understates_then_data_continues",
        "RIFF size ends after fmt but a data chunk follows anyway",
        False,
        True,
        lambda: _understated_riff(rate, channels, first_part, second_part),
    )

    cases.sort(key=lambda item: item.name)
    return cases


def _chunk_crosses_body(rate: int, channels: int, pcm: bytes) -> bytes:
    honest = build_wav(
        [
            _chunk(b"fmt ", _fmt_payload(rate, channels)),
            _chunk(b"data", pcm),
        ]
    )
    tampered = bytearray(honest)
    # For this canonical layout the data chunk size field sits at offset 40.
    tampered[40:44] = struct.pack("<I", len(pcm) + 256)
    return bytes(tampered)


def _hidden_tail(rate: int, channels: int, first_part: bytes, second_part: bytes) -> bytes:
    # RIFF size covers fmt + first data only; the second data is trailing
    # bytes that a boundary-respecting parser ignores and a boundary-less
    # parser silently consumes as replacement media.
    inner = build_wav(
        [
            _chunk(b"fmt ", _fmt_payload(rate, channels)),
            _chunk(b"data", first_part),
        ]
    )
    hidden = bytearray(b"data") + struct.pack("<I", len(second_part)) + second_part
    if len(second_part) % 2 == 1:
        hidden += b"\x00"
    return inner + bytes(hidden)


def _understated_riff(rate: int, channels: int, first_part: bytes, second_part: bytes) -> bytes:
    full = build_wav(
        [
            _chunk(b"fmt ", _fmt_payload(rate, channels)),
            _chunk(b"data", first_part),
            _chunk(b"data", second_part),
        ]
    )
    # Shrink the declared RIFF size to cover the WAVE form type and fmt chunk
    # alone. The boundary-respecting oracle then sees no data chunk and
    # rejects; the production parser walks the whole buffer regardless.
    fmt_body_size = 4 + 8 + 16  # form type + fmt header + fmt payload
    tampered = bytearray(full)
    tampered[4:8] = struct.pack("<I", fmt_body_size)
    return bytes(tampered)
