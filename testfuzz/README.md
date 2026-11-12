# Generative verification for source registration and rendering

The production smoke workflow only feeds one sine WAV and one blob of
non-audio bytes. That cannot find media which still looks like a WAV, is
accepted by the service, but whose stored facts no longer describe the
actual media. This package adds generative, differential verification built
around an **independent** judge and automatic, fact-preserving reduction.

## Layout

| File | Role |
|---|---|
| `oracle/wav_oracle.py` | From-scratch RIFF/WAVE reference parser and level metrics. No project code is imported; only the Python standard library is used. |
| `wav_generator.py` | Deterministic corpus builder: canonical media, legal-but-uncommon containers, structurally invalid media, and WAV-shaped containers that lie about their payload. |
| `service_client.py` | JSON-over-HTTP client restricted to the published API. |
| `verifier.py` | Compares the oracle verdict with observations made over real HTTP, including material facts and the render path. |
| `reducer.py` | Media-aware delta debugging; every smaller candidate is re-verified over the same HTTP path and must keep the original data error. |
| `run_fuzz.py`, `run.py` | CLI entry point and isolated-mode launcher. |

## Run

The runner boots the real service via `cmd/fuzzmedia` on an isolated data
directory and drives it entirely through HTTP:

```bash
python3 -I testfuzz/run.py
```

Use an already running service instead:

```bash
python3 -I testfuzz/run.py --base-url http://127.0.0.1:8093
```

Registration-only checks (skip timeline/render verification):

```bash
python3 -I testfuzz/run.py --no-render-checks
```

The process exits non-zero whenever any divergence is found. Findings land
in `testfuzz/findings/<case>/` with `original.wav`, the minimised
`reduced.wav`, a structured `case.json`, and a self-contained `replay.sh`
that reproduces the divergence over the same HTTP route.

## How the oracle stays independent

`wav_oracle.py` re-derives the RIFF/WAVE rules from the container format and
the service's published contract (16-bit integer PCM, mono/stereo,
8000-192000 Hz) instead of calling `internal/audiocodec`:

* chunk walks stop at the RIFF size declared in the header, so bytes outside
  the container cannot be treated as media;
* **every** `data` chunk inside the RIFF body contributes to the logical
  payload, so a later chunk can never silently replace an earlier one;
* legal but uncommon organisations are accepted: unknown chunks, LIST/INFO,
  odd-sized chunks with the standard pad byte, `data` before `fmt`, fmt
  extension bytes, WAVEFORMATEXTENSIBLE PCM, and split data chunks;
* genuinely invalid media is rejected with a structured reason (bad RIFF
  size, missing fmt/data, non-PCM tags, wrong width/channels, alignment
  lies, duplicate conflicting fmt chunks, chunks crossing the RIFF body).

The verifier never treats an atypical but legal structure as invalid.
Accept/reject disagreement and per-field fact disagreement
(`frame_count`, `duration_ms`, rate, channels, digest, independently
recomputed peak/RMS) are the only failure signals.

## Render-path verification

When both sides accept a source, the verifier creates a full-span timeline
and render over HTTP, then parses the downloaded output with the same
oracle and cross-checks the receipt. A registration-time fact error is
therefore observable again downstream: for example, a source whose frame
count was truncated at registration makes a timeline over the true frame
interval fail with `invalid_source_interval`, while a boundary-hidden tail
produces a render whose measured facts match the truncated record rather
than the real payload.

## Divergence categories

| Category | Meaning |
|---|---|
| `false_reject` | Oracle accepts legal media; the service returns an error. |
| `false_accept` | Oracle rejects invalid/ambiguous media; the service stores it. |
| `fact_mismatch` | Both accept, but stored frame count, duration, rate, channels, digest, or level measurements contradict the independent parse. |
| `reject_status` | Both reject, but the service does not use the stable 422 contract. |
| `partial_source_visible` | A failed registration leaves a readable source behind. |
| `readback_mismatch` / `readback_failed` | GET after PUT does not return the committed record. |
| `render_path_*` | Timeline, job, output, receipt, or output digest observed over HTTP contradicts the source truth. |

## Reduction guarantees

The reducer applies container-level transformations (shrink PCM to one
frame per data part, drop or shorten optional chunks, trim fmt extensions)
rather than deleting arbitrary bytes. A candidate is accepted only when:

1. the same divergence category remains observable through the same HTTP
   requests against a fresh verifier; and
2. for semantic cases, the oracle's fact signature survives - same number
   of data-bearing chunks, every part still non-empty, same channel/rate
   identity. Frame counts may shrink, but the hidden or replaced data that
   caused the original error cannot be removed, and a case cannot be
   reduced into a mere crash that loses the original data discrepancy.

Understated RIFF boundaries are recomputed during shrinking so the
boundary's meaning (which chunks the container actually claims) is
preserved rather than accidentally fixed.

## Production boundary

Nothing under `internal/` or the existing `cmd/` entry points is modified.
The only Go addition is the test-only `cmd/fuzzmedia` launcher, which wires
the same production storage, service, and HTTP layers onto an ephemeral
port and data directory.
