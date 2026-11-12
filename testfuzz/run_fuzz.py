"""Generative verification entry point.

The runner boots the real Go service (cmd/fuzzmedia) on an isolated data
directory, generates the structural WAV corpus, differentially verifies
registration and render behaviour against the independent oracle, and
automatically minimises every divergent case through the same HTTP path.

Artifacts for each divergence are written under testfuzz/findings:

* ``case.json``      - structured report including both fact sets;
* ``reduced.wav``    - the smallest container that still shows the
                       divergence over HTTP;
* ``original.wav``   - the pre-reduction container for replay comparison;
* ``replay.sh``      - a self-contained replay script using only curl and
                       base64 against any running service base URL.

Production decoding and business rules are never modified or imported by
the verification code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

from .reducer import Reducer
from .service_client import ServiceClient
from .verifier import DifferentialVerifier
from .wav_generator import generate_corpus

REPO_ROOT = Path(__file__).resolve().parent.parent
FINDINGS_DIR = REPO_ROOT / "testfuzz" / "findings"


def _resolve_go() -> str:
    """Locate the Go toolchain without assuming it is on the inherited PATH."""
    on_path = shutil.which("go")
    if on_path:
        return on_path
    candidates = [
        Path.home() / "tools" / "go" / "bin" / "go",
        Path("/usr/local/go/bin/go"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("Go toolchain not found; put 'go' on PATH or install to ~/tools/go")


class ServerProcess:
    def __init__(self, data_dir: Optional[Path]) -> None:
        self.data_dir = data_dir
        self.process: Optional[subprocess.Popen] = None
        self.base_url: Optional[str] = None
        self.binary: Optional[Path] = None

    def start(self) -> str:
        # Build once and run the binary directly. Using ``go run`` would
        # leave the compiled child reparented to init when the parent is
        # terminated, leaking servers across fuzz runs.
        go_binary = _resolve_go()
        self.binary = REPO_ROOT / "testfuzz" / ".bin" / "fuzzmedia"
        self.binary.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [go_binary, "build", "-buildvcs=false", "-o", str(self.binary), "./cmd/fuzzmedia"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as build_error:
            raise RuntimeError(
                f"failed to build cmd/fuzzmedia via {go_binary}: {build_error.stderr.strip()}"
            ) from build_error
        command = [str(self.binary)]
        if self.data_dir is not None:
            command.extend(["-data", str(self.data_dir)])
        env = dict(os.environ)
        go_bin_dir = str(Path(go_binary).parent)
        env["PATH"] = go_bin_dir + os.pathsep + env.get("PATH", "")
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        assert self.process.stdout is not None
        first_line = self.process.stdout.readline()
        if not first_line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"fuzzmedia exited before announcing its URL: {stderr}")
        parts = first_line.strip().split()
        if len(parts) != 2 or parts[0] != "BASE_URL":
            raise RuntimeError(f"unexpected fuzzmedia banner: {first_line!r}")
        self.base_url = parts[1]
        self._wait_for_health()
        return self.base_url

    def _wait_for_health(self, attempts: int = 30) -> None:
        client = ServiceClient(self.base_url or "")
        for _ in range(attempts):
            try:
                if client.health().status == 200:
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.5)
        raise RuntimeError("service did not become healthy")

    def stop(self) -> None:
        if self.process is None:
            return
        # Kill the whole process group so no listener survives the runner.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
        finally:
            self.process = None


def _verifier_factory(client: ServiceClient, prefix: str) -> Callable[[], DifferentialVerifier]:
    def factory() -> DifferentialVerifier:
        # A new verifier resets the per-verifier ID counter while keeping the
        # same network client and server.
        return DifferentialVerifier(client, id_prefix=prefix)

    return factory


def _write_replay_script(directory: Path, name: str) -> None:
    script = directory / "replay.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "# Replay the reduced media against a running voice-timeline service.\n"
        "# Usage: BASE_URL=http://127.0.0.1:8093 ./replay.sh\n"
        "set -euo pipefail\n"
        ": \"${BASE_URL:?set BASE_URL to a running voice-timeline service}\"\n"
        "cd \"$(dirname \"$0\")\"\n"
        f"SOURCE_ID=\"replay_{name}\"\n"
        "PAYLOAD=$(python3 -I -c \"import base64;print(base64.b64encode(open('reduced.wav','rb').read()).decode())\")\n"
        "curl -sS -X PUT \"$BASE_URL/v1/sources/$SOURCE_ID\" \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d \"{\\\"name\\\":\\\"replay\\\",\\\"wav_base64\\\":\\\"$PAYLOAD\\\"}\" | python3 -m json.tool\n"
        "echo\n"
        "curl -sS \"$BASE_URL/v1/sources/$SOURCE_ID\" | python3 -m json.tool\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def run(base_url: Optional[str], keep_artifacts: bool, render_checks: bool) -> int:
    corpus = generate_corpus()
    if FINDINGS_DIR.exists():
        shutil.rmtree(FINDINGS_DIR)
    FINDINGS_DIR.mkdir(parents=True)

    owns_server = base_url is None
    server: Optional[ServerProcess] = None
    if owns_server:
        server = ServerProcess(data_dir=None)
        base_url = server.start()
    client = ServiceClient(base_url)

    failures = 0
    try:
        verifier = DifferentialVerifier(client, id_prefix="gen")
        reports = []
        for case in corpus:
            wav = case.builder()
            report = verifier.verify_registration(
                case.name, case.description, case.expectation, case.semantic, wav
            )
            if report.ok and render_checks and report.oracle_accepted and report.service_accepted:
                verifier.verify_render_path(report, wav)
            reports.append((case, wav, report))

        passed = sum(1 for _, _, report in reports if report.ok)
        print(f"corpus cases: {len(reports)}  agreement: {passed}  divergence: {len(reports) - passed}")

        divergent = [(case, wav, report) for case, wav, report in reports if not report.ok]
        for case, original_wav, report in divergent:
            failures += 1
            print(f"DIVERGENCE {case.name}: " + "; ".join(f"{d.category} ({d.detail})" for d in report.divergences))
            reducer = Reducer(_verifier_factory(client, "red"))
            reduced = reducer.reduce(
                case.name, case.description, case.expectation, case.semantic, original_wav
            )
            case_dir = FINDINGS_DIR / case.name
            case_dir.mkdir(parents=True)
            (case_dir / "original.wav").write_bytes(original_wav)
            (case_dir / "reduced.wav").write_bytes(reduced.wav)
            payload = {
                "case": {
                    "name": case.name,
                    "description": case.description,
                    "oracle_expected_accept": case.expectation,
                    "semantic": case.semantic,
                },
                "original_size": len(original_wav),
                "reduced_size": reduced.reduced_size,
                "reduction_iterations": reduced.iterations,
                "preserved_oracle_facts": reduced.preserved_facts,
                "report": _serializable_report(reduced.report),
            }
            (case_dir / "case.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
            _write_replay_script(case_dir, case.name)
            print(
                f"  reduced {len(original_wav)} -> {reduced.reduced_size} bytes "
                f"in {reduced.iterations} trials; facts preserved: {reduced.preserved_facts}"
            )
    finally:
        if server is not None:
            server.stop()
        binary_dir = REPO_ROOT / "testfuzz" / ".bin"
        if binary_dir.exists():
            shutil.rmtree(binary_dir)
        for cache in REPO_ROOT.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        if failures == 0 and not keep_artifacts and FINDINGS_DIR.exists():
            shutil.rmtree(FINDINGS_DIR)

    print("RESULT " + ("FAIL" if failures else "PASS") + f" ({failures} divergent cases)")
    return 1 if failures else 0


def _serializable_report(report) -> dict:
    data = asdict(report)
    return data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generative WAV registration and render verification")
    parser.add_argument("--base-url", help="use an already running service instead of booting cmd/fuzzmedia")
    parser.add_argument("--keep-artifacts", action="store_true", help="keep the findings directory even on success")
    parser.add_argument("--no-render-checks", action="store_true", help="only verify registration facts")
    args = parser.parse_args(argv)
    return run(args.base_url, args.keep_artifacts, not args.no_render_checks)


if __name__ == "__main__":
    sys.exit(main())
