"""Differential verifier between the independent oracle and the live service.

For every generated container the verifier:

1. asks the independent RIFF/WAVE oracle what the media actually is;
2. submits the exact same bytes through the public HTTP registration path;
3. when both accept, compares every material fact the service later relies
   on for editing and rendering (frame count, rate, channels, duration and
   independently recomputed level measurements);
4. when the oracle accepts a semantic container, drives a timeline and
   render through HTTP and checks the render output with the same oracle,
   so a registration-time fact error cannot hide behind a later stage.

A divergence is recorded with an explicit category. Accept-only verdicts
never treat an uncommon but legal structure as invalid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .oracle import wav_oracle
from .service_client import ServiceClient

# Numeric tolerances. Float32 quantization inside the service is expected;
# structural facts must match exactly.
PEAK_TOLERANCE_DB = 0.5
RMS_TOLERANCE_DB = 0.5
LEVEL_FLOOR = wav_oracle.SILENCE_FLOOR_DBFS + 0.01


@dataclass
class Divergence:
    category: str
    detail: str


@dataclass
class CaseReport:
    name: str
    description: str
    expectation_accept: bool
    semantic: bool
    oracle_accepted: bool
    oracle_reason: Optional[str]
    http_status: int
    service_accepted: bool
    service_error_code: Optional[str]
    divergences: List[Divergence] = field(default_factory=list)
    # Facts observed independently and over HTTP, retained for reporting.
    oracle_facts: Dict[str, Any] = field(default_factory=dict)
    service_facts: Dict[str, Any] = field(default_factory=dict)
    render_checked: bool = False
    render_detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.divergences


def _error_code(body: bytes) -> Optional[str]:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except ValueError:
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    return error.get("code") if isinstance(error, dict) else None


def _duration_ms(frame_count: int, sample_rate: int) -> int:
    if frame_count <= 0 or sample_rate <= 0:
        return 0
    return int(frame_count * 1000 / sample_rate)


def _levels_close(measured: float, observed: float, tolerance: float) -> bool:
    # Both silent: floors can differ by a few ULPs and must not diverge.
    if measured <= LEVEL_FLOOR and observed <= LEVEL_FLOOR:
        return True
    return abs(measured - observed) <= tolerance


class DifferentialVerifier:
    # Shared across instances so every HTTP registration inside one fuzz run
    # gets a unique resource ID; otherwise the service's idempotent PUT would
    # answer a reduction trial with an earlier case's stored record.
    _global_counter = 0

    def __init__(self, client: ServiceClient, id_prefix: str = "fz") -> None:
        self.client = client
        self.id_prefix = id_prefix
        self._counter = 0

    def _next_id(self, label: str) -> str:
        DifferentialVerifier._global_counter += 1
        self._counter = DifferentialVerifier._global_counter
        # Resource IDs allow letters, digits, underscore, hyphen and must be
        # 3-64 characters. Corpus names only use those characters anyway.
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)[:40]
        return f"{self.id_prefix}_{self._counter:05d}_{safe}"

    def verify_registration(self, name: str, description: str, expectation: bool, semantic: bool, wav: bytes) -> CaseReport:
        verdict = wav_oracle.parse_wav(wav)
        digest = hashlib.sha256(wav).hexdigest()
        source_id = self._next_id(name)
        result = self.client.put_source(source_id, f"fuzz {name}", wav)
        service_accepted = result.status in (200, 201)
        service_error = _error_code(result.body)

        report = CaseReport(
            name=name,
            description=description,
            expectation_accept=expectation,
            semantic=semantic,
            oracle_accepted=verdict.accepted,
            oracle_reason=verdict.reject_reason,
            http_status=result.status,
            service_accepted=service_accepted,
            service_error_code=service_error,
        )

        if verdict.accepted:
            report.oracle_facts = self._oracle_facts(verdict, digest, len(wav))
        if service_accepted:
            report.service_facts = result.json()

        # Primary accept/reject differential.
        if verdict.accepted and not service_accepted:
            report.divergences.append(
                Divergence("false_reject", f"oracle accepts legal container, service returns {result.status} ({service_error})")
            )
            return report
        if not verdict.accepted and service_accepted:
            report.divergences.append(
                Divergence(
                    "false_accept",
                    f"oracle rejects ({verdict.reject_reason}: {verdict.reject_detail}), service returns {result.status}",
                )
            )
            return report
        if not verdict.accepted and not service_accepted:
            if result.status != 422:
                report.divergences.append(
                    Divergence("reject_status", f"invalid media rejected with status {result.status}, want 422")
                )
            # A failed registration must leave no visible partial source.
            read_back = self.client.get_source(source_id)
            if read_back.status != 404:
                report.divergences.append(
                    Divergence("partial_source_visible", f"failed registration readable with status {read_back.status}")
                )
            return report

        # Both accept: compare the material facts end to end.
        self._compare_facts(report, verdict, digest, len(wav))
        return report

    def verify_render_path(self, report: CaseReport, wav: bytes) -> None:
        """Drive an accepted source through timeline + render over HTTP."""
        if not report.service_accepted or not report.oracle_accepted:
            return
        source_id = report.service_facts.get("id")
        if not source_id:
            return
        oracle_frames = report.oracle_facts["frame_count"]
        # Select the full source interval; timeline validation uses the
        # service's own (possibly wrong) frame count, which is part of the
        # observable behaviour under test.
        timeline_id = self._next_id("tl_" + report.name)
        timeline = self.client.put_timeline(
            timeline_id,
            f"fuzz timeline {report.name}",
            sample_rate=report.oracle_facts["sample_rate"],
            output_channels=report.oracle_facts["channels"],
            target_peak_dbfs=-1.0,
            spans=[
                {
                    "source_id": source_id,
                    "source_start_frame": 0,
                    "source_end_frame": oracle_frames,
                    "gain_db": 0.0,
                    "crossfade_frames": 0,
                }
            ],
        )
        if timeline.status not in (200, 201):
            report.divergences.append(
                Divergence("render_path_timeline_rejected", f"timeline over accepted source returned {timeline.status}: {timeline.body.decode('utf-8', 'replace')[:200]}")
            )
            report.render_checked = True
            report.render_detail = "timeline rejected"
            return

        render_id = self._next_id("rd_" + report.name)
        render = self.client.put_render(render_id, timeline_id)
        if render.status not in (200, 201):
            report.divergences.append(
                Divergence("render_path_render_rejected", f"render returned {render.status}: {render.body.decode('utf-8', 'replace')[:200]}")
            )
            report.render_checked = True
            report.render_detail = "render rejected"
            return
        job = render.json()
        if job.get("state") != "completed":
            report.divergences.append(
                Divergence("render_path_not_completed", f"job state is {job.get('state')} with error {job.get('error')}")
            )
            report.render_checked = True
            report.render_detail = f"state {job.get('state')}"
            return

        output_result = self.client.get_output(render_id)
        if output_result.status != 200:
            report.divergences.append(
                Divergence("render_path_output_unreadable", f"output fetch returned {output_result.status}")
            )
            report.render_checked = True
            report.render_detail = "output unreadable"
            return

        receipt_result = self.client.get_receipt(render_id)
        output_verdict = wav_oracle.parse_wav(output_result.body)
        if not output_verdict.accepted:
            report.divergences.append(
                Divergence(
                    "render_output_invalid",
                    f"oracle rejects rendered WAV: {output_verdict.reject_reason} {output_verdict.reject_detail}",
                )
            )
            report.render_checked = True
            report.render_detail = "output invalid"
            return

        receipt = receipt_result.json()
        receipt_frames = receipt["output"]["frame_count"]
        if output_verdict.frame_count != receipt_frames:
            report.divergences.append(
                Divergence(
                    "render_receipt_frame_lie",
                    f"output actually has {output_verdict.frame_count} frames, receipt claims {receipt_frames}",
                )
            )
        if output_verdict.frame_count != oracle_frames:
            report.divergences.append(
                Divergence(
                    "render_frame_count_lie",
                    f"full-span render produced {output_verdict.frame_count} frames, source truth is {oracle_frames}",
                )
            )
        output_digest = hashlib.sha256(output_result.body).hexdigest()
        if output_digest != receipt["output"]["digest"]:
            report.divergences.append(Divergence("render_digest_lie", "output digest does not match receipt"))
        report.render_checked = True
        report.render_detail = f"rendered {output_verdict.frame_count} frames"

    def _oracle_facts(self, verdict: wav_oracle.WaveVerdict, digest: str, byte_count: int) -> Dict[str, Any]:
        peak, rms = wav_oracle.measure(verdict.samples)
        return {
            "digest": digest,
            "byte_count": byte_count,
            "sample_rate": verdict.sample_rate,
            "channels": verdict.channels,
            "bits_per_sample": verdict.bits_per_sample,
            "frame_count": verdict.frame_count,
            "duration_ms": _duration_ms(verdict.frame_count, verdict.sample_rate),
            "peak_dbfs": peak,
            "rms_dbfs": rms,
            "data_chunk_count": verdict.data_chunk_count,
        }

    def _compare_facts(self, report: CaseReport, verdict: wav_oracle.WaveVerdict, digest: str, byte_count: int) -> None:
        facts = self._oracle_facts(verdict, digest, byte_count)
        observed = report.service_facts
        exact_fields = (
            "digest",
            "byte_count",
            "sample_rate",
            "channels",
            "bits_per_sample",
            "frame_count",
            "duration_ms",
        )
        for field_name in exact_fields:
            if observed.get(field_name) != facts[field_name]:
                report.divergences.append(
                    Divergence(
                        "fact_mismatch",
                        f"{field_name}: service={observed.get(field_name)!r} oracle={facts[field_name]!r}",
                    )
                )
        if not _levels_close(facts["peak_dbfs"], float(observed.get("peak_dbfs", 0.0)), PEAK_TOLERANCE_DB):
            report.divergences.append(
                Divergence(
                    "fact_mismatch",
                    f"peak_dbfs: service={observed.get('peak_dbfs')!r} oracle={facts['peak_dbfs']:.3f}",
                )
            )
        if not _levels_close(facts["rms_dbfs"], float(observed.get("rms_dbfs", 0.0)), RMS_TOLERANCE_DB):
            report.divergences.append(
                Divergence(
                    "fact_mismatch",
                    f"rms_dbfs: service={observed.get('rms_dbfs')!r} oracle={facts['rms_dbfs']:.3f}",
                )
            )
        # Read-back must return the identical record; a dropped fact here is
        # observable to every later editing client.
        read_back = self.client.get_source(observed.get("id", ""))
        if read_back.status != 200:
            report.divergences.append(Divergence("readback_failed", f"GET source returned {read_back.status}"))
            return
        stored = read_back.json()
        for field_name in exact_fields:
            if stored.get(field_name) != observed.get(field_name):
                report.divergences.append(
                    Divergence(
                        "readback_mismatch",
                        f"{field_name}: write response={observed.get(field_name)!r} read-back={stored.get(field_name)!r}",
                    )
                )
