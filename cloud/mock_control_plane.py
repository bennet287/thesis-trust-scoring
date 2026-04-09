import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ControlPlaneHandler(BaseHTTPRequestHandler):
    manifest = None
    package_bytes = None

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, payload, content_type="application/octet-stream", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.rstrip("/")

        if path.startswith("/check/"):
            version = self.manifest["version"]
            payload = {
                "available": True,
                "version": version,
                "hash": self.manifest["hash"],
                "size": self.manifest["size"],
                "manifest_url": f"http://{self.server.server_address[0]}:{self.server.server_address[1]}/manifest/{version}",
                "package_url": f"http://{self.server.server_address[0]}:{self.server.server_address[1]}/download/{version}",
            }
            self._send_json(payload)
            return

        if path == f"/manifest/{self.manifest['version']}":
            self._send_json(self.manifest)
            return

        if path == f"/download/{self.manifest['version']}":
            self._send_bytes(self.package_bytes, content_type="application/gzip")
            return

        self._send_json({"error": "not found"}, status=404)

    def log_message(self, fmt, *args):
        print("[mock-control-plane] " + fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local mock control plane for sync client demo")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--package", required=True, help="Path to package tar.gz")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    package_bytes = Path(args.package).read_bytes()

    ControlPlaneHandler.manifest = manifest
    ControlPlaneHandler.package_bytes = package_bytes

    server = ThreadingHTTPServer((args.host, args.port), ControlPlaneHandler)
    print(f"Serving mock control plane at http://{args.host}:{args.port}")
    print(f"Version: {manifest['version']}")
    server.serve_forever()


if __name__ == "__main__":
    main()
