"""Minimal JSON-over-HTTP client for the voice timeline service.

The client only knows the published HTTP contract. It never imports
production Go code and never reads the service data directory directly, so
the verifier's notion of truth stays independent of the decoder under test.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HTTPResult:
    status: int
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class ServiceClient:
    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]]) -> HTTPResult:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return HTTPResult(response.status, response.read())
        except urllib.error.HTTPError as error:
            return HTTPResult(error.code, error.read())

    def health(self) -> HTTPResult:
        return self._request("GET", "/healthz", None)

    def put_source(self, source_id: str, name: str, wav: bytes) -> HTTPResult:
        return self._request(
            "PUT",
            f"/v1/sources/{source_id}",
            {"name": name, "wav_base64": base64.b64encode(wav).decode("ascii")},
        )

    def get_source(self, source_id: str) -> HTTPResult:
        return self._request("GET", f"/v1/sources/{source_id}", None)

    def put_timeline(
        self,
        timeline_id: str,
        name: str,
        sample_rate: int,
        output_channels: int,
        target_peak_dbfs: float,
        spans: List[Dict[str, Any]],
    ) -> HTTPResult:
        return self._request(
            "PUT",
            f"/v1/timelines/{timeline_id}",
            {
                "name": name,
                "sample_rate": sample_rate,
                "output_channels": output_channels,
                "target_peak_dbfs": target_peak_dbfs,
                "spans": spans,
            },
        )

    def put_render(self, render_id: str, timeline_id: str) -> HTTPResult:
        return self._request("PUT", f"/v1/renders/{render_id}", {"timeline_id": timeline_id})

    def get_render(self, render_id: str) -> HTTPResult:
        return self._request("GET", f"/v1/renders/{render_id}", None)

    def get_receipt(self, render_id: str) -> HTTPResult:
        return self._request("GET", f"/v1/renders/{render_id}/receipt", None)

    def get_output(self, render_id: str) -> HTTPResult:
        return self._request("GET", f"/v1/renders/{render_id}/output", None)
