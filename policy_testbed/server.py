"""Local HTTP authorization testbed (not production identity infrastructure).

Unlike a classifier returning ALLOW/BLOCK labels, this server truly prevents
HTTP reads and writes unless a verified token's principal has permission.
It binds to loopback; no AI risk scores are used or trusted at this stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping
from urllib.parse import urlsplit

READ = ("GET", "/v1/telemetry")
WRITE = ("POST", "/v1/config")
PERMISSIONS: dict[str, frozenset[tuple[str, str]]] = {
    "sensor-A": frozenset({READ}),
    "operator-B": frozenset({READ, WRITE}),
}


class TokenRegistry:
    """Demonstration bearer credentials, stored as token digests in memory."""

    def __init__(self, tokens_by_principal: Mapping[str, str]):
        if not tokens_by_principal:
            raise ValueError("No authentication credentials configured; refusing to start")
        if len(set(tokens_by_principal.values())) != len(tokens_by_principal):
            raise ValueError("Tokens must not be shared across principals")
        if any(not principal or not token or len(token) < 16 for principal, token in tokens_by_principal.items()):
            raise ValueError("All principals need nonempty identifiers and >=16 character random tokens")
        self._digests = {
            principal: hashlib.sha256(token.encode("utf-8")).digest()
            for principal, token in tokens_by_principal.items()
        }

    def authenticate(self, header: str | None) -> str | None:
        if not header or not header.startswith("Bearer "):
            return None
        token = header[len("Bearer "):]
        if not token or token.strip() != token:
            return None
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for principal, expected in self._digests.items():
            if secrets.compare_digest(digest, expected):
                return principal
        return None


@dataclass
class TestbedState:
    tokens: TokenRegistry
    sampling_interval: int = 30
    audit: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def authorized(self, principal: str, method: str, path: str) -> bool:
        return (method, path) in PERMISSIONS.get(principal, frozenset())

    def record(self, *, principal: str | None, method: str, path: str, decision: str) -> None:
        # Deliberately omit bearer tokens and request payloads from audit events.
        with self.lock:
            self.audit.append({"principal": principal, "method": method, "resource": path, "decision": decision})


def make_server(state: TestbedState, port: int = 0) -> ThreadingHTTPServer:
    """Return server bound to localhost only, with no network exposure."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # suppress request headers and credentials in console logs

        def json_response(self, code: int, data: dict) -> None:
            body = json.dumps(data, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            if code == 401:
                self.send_header("WWW-Authenticate", 'Bearer realm="edge-testbed"')
            self.end_headers()
            self.wfile.write(body)

        def enforce(self, method: str, path: str) -> str | None:
            principal = state.tokens.authenticate(self.headers.get("Authorization"))
            if principal is None:
                state.record(principal=None, method=method, path=path, decision="unauthenticated")
                self.json_response(401, {"error": "authentication_required"})
                return None
            if not state.authorized(principal, method, path):
                state.record(principal=principal, method=method, path=path, decision="denied_by_policy")
                self.json_response(403, {"error": "permission_denied"})
                return None
            return principal

        def route(self, method: str) -> None:
            # Resource and action are taken from the HTTP request, never an untrusted claim.
            parsed = urlsplit(self.path)
            path = parsed.path
            if parsed.query or parsed.fragment or (method, path) not in (READ, WRITE):
                state.record(principal=None, method=method, path=path, decision="unknown_resource")
                self.json_response(404, {"error": "not_found"})
                return
            principal = self.enforce(method, path)
            if principal is None:
                return
            if (method, path) == READ:
                with state.lock:
                    interval = state.sampling_interval
                state.record(principal=principal, method=method, path=path, decision="allowed")
                self.json_response(200, {"sensor": "temperature", "celsius": 22.5, "sampling_interval": interval})
                return
            # Authorized configuration write, with strict JSON shape and value validation.
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 4096:
                    raise ValueError("invalid_length")
                payload = json.loads(self.rfile.read(size))
                if (not isinstance(payload, dict) or set(payload) != {"sampling_interval"}
                        or type(payload["sampling_interval"]) is not int
                        or not 1 <= payload["sampling_interval"] <= 3600):
                    raise ValueError("invalid_payload")
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                state.record(principal=principal, method=method, path=path, decision="invalid_payload")
                self.json_response(400, {"error": "invalid_payload"})
                return
            with state.lock:
                state.sampling_interval = payload["sampling_interval"]
            state.record(principal=principal, method=method, path=path, decision="allowed")
            self.json_response(200, {"updated": True, "sampling_interval": payload["sampling_interval"]})

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    raw = os.environ.get("EDGE_TESTBED_TOKENS_JSON")
    if not raw:
        parser.error("Provide EDGE_TESTBED_TOKENS_JSON with principal-to-token mapping; never commit credentials")
    try:
        credentials = json.loads(raw)
        if not isinstance(credentials, dict):
            raise ValueError("expected JSON object")
        state = TestbedState(TokenRegistry(credentials))
    except (ValueError, TypeError) as error:
        parser.error(f"invalid token configuration: {error}")
    with make_server(state, args.port) as server:
        print(f"Local authorization testbed listening on http://127.0.0.1:{server.server_port}; no ML risk checks enabled")
        server.serve_forever()


if __name__ == "__main__":
    main()
