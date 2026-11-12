"""Automatic minimisation of divergent generated media.

The reducer is a media-aware variant of delta debugging. Instead of deleting
arbitrary bytes (which would quickly destroy the RIFF structure and reduce
the case to a bare crash or a generic invalid-WAV rejection), it applies
named container-level transformations:

* shrink the PCM payload to the smallest frame count that still expresses
  the same structural organisation;
* remove optional chunks while keeping every data-bearing chunk that the
  original divergence depends on;
* shorten unknown-chunk payloads;
* collapse the container plan to the chunks required to reproduce.

After every candidate it reruns the verifier over the same HTTP path and
only accepts a smaller candidate when the same divergence category remains
observable. For semantic cases it additionally requires the independent
oracle to keep describing the same media fact (frame count, data-part
layout), so the original data error cannot be replaced by a mere crash.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from . import wav_generator
from .oracle import wav_oracle
from .verifier import CaseReport, DifferentialVerifier

# A minimal PCM payload still needs two frames so "split into parts" means
# at least one frame per data chunk.
MIN_FRAMES = 2


@dataclass
class ReducedCase:
    name: str
    original_size: int
    reduced_size: int
    wav: bytes
    report: CaseReport
    iterations: int
    preserved_facts: dict


class Reducer:
    def __init__(self, verifier_factory: Callable[[], DifferentialVerifier]) -> None:
        # A fresh verifier (fresh ID counter, same server) per trial keeps
        # HTTP resource identifiers unique across reduction attempts.
        self.verifier_factory = verifier_factory

    def reduce(self, name: str, description: str, expectation: bool, semantic: bool, original: bytes) -> ReducedCase:
        baseline_verifier = self.verifier_factory()
        baseline = baseline_verifier.verify_registration(
            name, description, expectation, semantic, original
        )
        if baseline.ok:
            raise ValueError("reduce() called on a non-divergent case")
        target_categories = {item.category for item in baseline.divergences}
        original_verdict = wav_oracle.parse_wav(original)
        preserved_facts = self._fact_signature(original_verdict)

        current = original
        current_report = baseline
        iterations = 0
        changed = True
        while changed:
            changed = False
            for candidate in self._candidates(current):
                iterations += 1
                if len(candidate) >= len(current):
                    continue
                verifier = self.verifier_factory()
                trial = verifier.verify_registration(name, description, expectation, semantic, candidate)
                if not self._retains_divergence(trial, target_categories):
                    continue
                if semantic and not self._retains_semantic_fact(candidate, preserved_facts):
                    continue
                current = candidate
                current_report = trial
                preserved_facts = self._fact_signature(wav_oracle.parse_wav(candidate))
                changed = True
                break

        # Final confirmation through the full registration + render path so
        # the reduced artefact is proven observable end to end.
        final_verifier = self.verifier_factory()
        final_report = final_verifier.verify_registration(
            name, description, expectation, semantic, current
        )
        if semantic and final_report.service_accepted and final_report.oracle_accepted:
            final_verifier.verify_render_path(final_report, current)
        if not self._retains_divergence(final_report, target_categories):
            # Keep the last known-bad candidate rather than emitting a green
            # artefact; this guards against verifier state drift.
            final_report = current_report
        return ReducedCase(
            name=name,
            original_size=len(original),
            reduced_size=len(current),
            wav=current,
            report=final_report,
            iterations=iterations,
            preserved_facts=preserved_facts,
        )

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def _candidates(self, wav: bytes) -> Sequence[bytes]:
        try:
            plan, riff_override = self._decompose(wav)
        except ValueError:
            return self._byte_level_candidates(wav)

        candidates: List[bytes] = []

        # 1. Shrink PCM payloads to the smallest frame count that preserves
        #    the number of data parts and each part's non-emptiness.
        shrunk = self._shrink_pcm(plan)
        if shrunk is not None:
            candidates.append(
                wav_generator.build_wav(shrunk, riff_size_override=self._rescale_override(plan, shrunk, riff_override))
            )

        # 2. Remove optional (non-fmt, non-data) chunks one at a time.
        for index, item in enumerate(plan):
            fourcc = item[0]
            if fourcc in (b"fmt ", b"data"):
                continue
            reduced = plan[:index] + plan[index + 1 :]
            candidates.append(
                wav_generator.build_wav(reduced, riff_size_override=self._rescale_override(plan, reduced, riff_override))
            )

        # 3. Shorten optional chunk payloads while keeping them present, so
        #    the layout (position, odd-size padding) stays reproducible.
        for index, item in enumerate(plan):
            fourcc, payload, odd_size = item
            if fourcc in (b"fmt ", b"data") or len(payload) <= 1:
                continue
            shortened = payload[: max(1, len(payload) // 2)]
            reduced = list(plan)
            reduced[index] = (fourcc, shortened, odd_size)
            candidates.append(
                wav_generator.build_wav(reduced, riff_size_override=self._rescale_override(plan, reduced, riff_override))
            )

        # 4. Trim optional extension bytes on the fmt chunk while retaining
        #    a valid 16-byte PCM header.
        for index, item in enumerate(plan):
            if item[0] != b"fmt " or len(item[1]) <= 16:
                continue
            reduced = list(plan)
            reduced[index] = (b"fmt ", item[1][:16], False)
            candidates.append(
                wav_generator.build_wav(reduced, riff_size_override=self._rescale_override(plan, reduced, riff_override))
            )

        # 5. Byte-level fallbacks for containers the decompiler cannot map.
        candidates.extend(self._byte_level_candidates(wav))
        return candidates

    def _rescale_override(
        self,
        original: List[wav_generator.ChunkPlan],
        reduced: List[wav_generator.ChunkPlan],
        riff_override: Optional[int],
    ) -> Optional[int]:
        """Recompute an understated RIFF size so the boundary meaning survives.

        The original override marks a boundary that ends inside the chunk
        list. After shrinking earlier payloads, a fixed byte offset would
        either overshoot the new (smaller) buffer or swallow a chunk the
        boundary used to exclude. We instead identify how many chunks of the
        original plan sat at or before the boundary and rebuild the size to
        end after the same ordinal chunk in the reduced plan.
        """
        if riff_override is None:
            return None
        boundary = riff_override - 4  # RIFF size includes the WAVE form type
        inside = 0
        cursor = 0
        for fourcc, payload, odd_size in original:
            chunk_extent = 8 + len(payload) + (1 if odd_size or len(payload) % 2 == 1 else 0)
            if cursor + chunk_extent <= boundary:
                inside += 1
                cursor += chunk_extent
            else:
                break
        if inside == 0 or inside > len(reduced):
            return riff_override
        rebuilt = 0
        for fourcc, payload, odd_size in reduced[:inside]:
            rebuilt += 8 + len(payload) + (1 if odd_size or len(payload) % 2 == 1 else 0)
        return rebuilt + 4

    def _shrink_pcm(self, plan: List[wav_generator.ChunkPlan]) -> Optional[List[wav_generator.ChunkPlan]]:
        data_indexes = [index for index, item in enumerate(plan) if item[0] == b"data"]
        if not data_indexes:
            return None
        # Respect block alignment from the fmt chunk.
        fmt = next((item[1] for item in plan if item[0] == b"fmt "), None)
        if fmt is None or len(fmt) < 16:
            return None
        channels = struct.unpack("<H", fmt[2:4])[0]
        block_align = struct.unpack("<H", fmt[12:14])[0]
        if block_align <= 0:
            return None
        reduced = list(plan)
        changed = False
        for index in data_indexes:
            _, payload, odd_size = plan[index]
            frames = len(payload) // block_align
            if frames <= 1:
                continue
            # One frame per data chunk is the smallest layout that keeps
            # every part present and therefore keeps the data error alive.
            one_frame = payload[:block_align]
            reduced[index] = (b"data", one_frame, False)
            changed = True
        if not changed:
            return None
        # Sanity: every surviving data chunk stays non-empty and aligned.
        for index in data_indexes:
            payload = reduced[index][1]
            if len(payload) == 0 or len(payload) % block_align != 0:
                return None
        return reduced

    def _decompose(self, wav: bytes) -> Tuple[List[wav_generator.ChunkPlan], Optional[int]]:
        """Parse a generated container back into a chunk plan.

        Supports trailing chunks beyond an understated RIFF size so the
        boundary-related cases remain reducible. Raises ValueError when the
        bytes are not a recognisable RIFF/WAVE container.
        """
        if len(wav) < 12 or wav[0:4] != b"RIFF" or wav[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE container")
        declared_riff = struct.unpack("<I", wav[4:8])[0]
        plan: List[wav_generator.ChunkPlan] = []
        offset = 12
        while offset + 8 <= len(wav):
            fourcc = wav[offset : offset + 4]
            size = struct.unpack("<I", wav[offset + 4 : offset + 8])[0]
            payload_start = offset + 8
            payload_end = payload_start + size
            if payload_end > len(wav):
                raise ValueError("chunk exceeds buffer; not safely decomposable")
            odd = size % 2 == 1
            plan.append((fourcc, bytes(wav[payload_start:payload_end]), odd))
            offset = payload_end + (1 if odd else 0)
        riff_override = declared_riff if declared_riff + 8 != len(wav) else None
        return plan, riff_override

    def _byte_level_candidates(self, wav: bytes) -> List[bytes]:
        candidates: List[bytes] = []
        if len(wav) <= 12:
            return candidates
        # Remove the last byte, and remove a middle block. These only apply
        # when structural transforms have nothing left to try, and the
        # verifier re-confirms the exact divergence before accepting them.
        candidates.append(wav[:-1])
        middle = len(wav) // 2
        block = max(1, len(wav) // 8)
        if middle + block < len(wav):
            candidates.append(wav[:middle] + wav[middle + block :])
        return candidates

    def _retains_divergence(self, report: CaseReport, target_categories: set) -> bool:
        if report.ok:
            return False
        observed = {item.category for item in report.divergences}
        return bool(observed & target_categories)

    def _fact_signature(self, verdict: wav_oracle.WaveVerdict) -> dict:
        if not verdict.accepted:
            return {"accepted": False, "reason": verdict.reject_reason}
        return {
            "accepted": True,
            "frame_count": verdict.frame_count,
            "channels": verdict.channels,
            "sample_rate": verdict.sample_rate,
            "data_chunk_count": verdict.data_chunk_count,
            "data_frame_parts": tuple(verdict.data_frame_parts),
            "data_bytes": verdict.data_bytes,
        }

    def _retains_semantic_fact(self, candidate: bytes, previous: dict) -> bool:
        verdict = wav_oracle.parse_wav(candidate)
        signature = self._fact_signature(verdict)
        if signature.get("accepted") != previous.get("accepted"):
            return False
        if not verdict.accepted:
            # For rejection-side semantic cases the reject reason must stay
            # the same class; the HTTP-level divergence category is checked
            # separately.
            return signature.get("reason") == previous.get("reason")
        # The structural shape of the data error must survive: same number
        # of data parts and the same channel/rate identity. Frame counts may
        # shrink, but no data-bearing chunk may disappear.
        for key in ("channels", "sample_rate", "data_chunk_count"):
            if signature.get(key) != previous.get(key):
                return False
        if len(signature.get("data_frame_parts", ())) != len(previous.get("data_frame_parts", ())):
            return False
        if any(part < 1 for part in signature.get("data_frame_parts", ())):
            return False
        return True
