"""Generative verification toolkit for the voice timeline renderer.

Modules:
* oracle.wav_oracle - independent RIFF/WAVE reference parser and metrics;
* wav_generator     - deterministic corpus of legal, uncommon, and lying
                      containers;
* service_client    - HTTP client restricted to the published API;
* verifier          - differential fact comparison over HTTP;
* reducer           - media-aware delta debugging of divergent cases;
* run_fuzz          - CLI entry point.
"""
