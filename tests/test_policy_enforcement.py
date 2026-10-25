"""Integration tests send real HTTP requests through the enforcement point."""
from __future__ import annotations

import json
import secrets
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from policy_testbed.server import TestbedState, TokenRegistry, make_server


class HTTPAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sensor_token = secrets.token_urlsafe(32)
        cls.operator_token = secrets.token_urlsafe(32)
        cls.unknown_token = secrets.token_urlsafe(32)
        cls.state = TestbedState(TokenRegistry({"sensor-A": cls.sensor_token, "operator-B": cls.operator_token}))
        cls.server = make_server(cls.state, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        with self.state.lock:
            self.state.sampling_interval = 30
            self.state.audit.clear()

    def request(self, method, path, token=None, payload=None, headers=None):
        req_headers = dict(headers or {})
        if token is not None:
            req_headers["Authorization"] = f"Bearer {token}"
        if payload is not None:
            req_headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(self.base_url + path, data=data, method=method, headers=req_headers)
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_sensor_can_read_telemetry(self):
        status, data = self.request("GET", "/v1/telemetry", self.sensor_token)
        self.assertEqual(status, 200)
        self.assertEqual(data["sampling_interval"], 30)

    def test_sensor_cannot_modify_configuration(self):
        status, _ = self.request("POST", "/v1/config", self.sensor_token, {"sampling_interval": 5})
        self.assertEqual(status, 403)
        self.assertEqual(self.state.sampling_interval, 30)
        self.assertEqual(self.state.audit[-1]["decision"], "denied_by_policy")

    def test_operator_can_modify_configuration(self):
        status, data = self.request("POST", "/v1/config", self.operator_token, {"sampling_interval": 5})
        self.assertEqual(status, 200)
        self.assertTrue(data["updated"])
        self.assertEqual(self.state.sampling_interval, 5)

    def test_missing_and_invalid_tokens_cannot_read(self):
        self.assertEqual(self.request("GET", "/v1/telemetry")[0], 401)
        self.assertEqual(self.request("GET", "/v1/telemetry", self.unknown_token)[0], 401)

    def test_spoofed_identity_header_does_not_grant_permission(self):
        status, _ = self.request("POST", "/v1/config", self.sensor_token, {"sampling_interval": 1}, {"X-Principal": "operator-B"})
        self.assertEqual(status, 403)
        self.assertEqual(self.state.sampling_interval, 30)

    def test_forbidden_path_and_query_cannot_bypass_policy(self):
        self.assertEqual(self.request("GET", "/v1/config", self.operator_token)[0], 404)
        self.assertEqual(self.request("POST", "/v1/config?override=true", self.operator_token, {"sampling_interval": 5})[0], 404)
        self.assertEqual(self.state.sampling_interval, 30)

    def test_invalid_payload_does_not_mutate_state(self):
        self.assertEqual(self.request("POST", "/v1/config", self.operator_token, {"sampling_interval": -1})[0], 400)
        self.assertEqual(self.request("POST", "/v1/config", self.operator_token, {"sampling_interval": True})[0], 400)
        self.assertEqual(self.state.sampling_interval, 30)

    def test_audit_excludes_credentials(self):
        self.request("POST", "/v1/config", self.sensor_token, {"sampling_interval": 1})
        content = json.dumps(self.state.audit)
        self.assertNotIn(self.sensor_token, content)
        self.assertNotIn(self.operator_token, content)
        self.assertEqual(self.state.audit[-1]["principal"], "sensor-A")

    def test_refuse_missing_credentials_and_duplicated_token(self):
        with self.assertRaises(ValueError):
            TokenRegistry({})
        with self.assertRaises(ValueError):
            TokenRegistry({"sensor-A": "same-token-value-123", "operator-B": "same-token-value-123"})


if __name__ == "__main__":
    unittest.main()
