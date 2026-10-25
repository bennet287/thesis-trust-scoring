"""Verify policy-first ML-risk *interface* over real localhost HTTP.

All scores are deterministic SERVER-INJECTED placeholders for the baseline.
This does not yet claim live trained-model inference or safety improvement.
"""
from __future__ import annotations

import json
import secrets
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from policy_testbed.risk_policy import decide
from policy_testbed.server import TestbedState, TokenRegistry, make_server


class RiskDecisionUnitTests(unittest.TestCase):
    def test_low_risk_does_not_bypass_explicit_denial(self):
        self.assertEqual(decide(False, mode="risk_gated", risk_score=0.0), "denied_by_policy")

    def test_policy_only_does_not_need_risk_evidence(self):
        self.assertEqual(decide(True, mode="policy_only"), "allowed")

    def test_below_threshold_allowed_and_equal_threshold_denied(self):
        self.assertEqual(decide(True, mode="risk_gated", risk_score=0.69), "allowed")
        self.assertEqual(decide(True, mode="risk_gated", risk_score=0.70), "denied_by_risk")

    def test_missing_invalid_nonfinite_risk_fails_closed(self):
        for value in (None, float("nan"), float("inf"), -0.1, 1.1, "invalid", True):
            with self.subTest(value=value):
                self.assertEqual(decide(True, mode="risk_gated", risk_score=value), "unavailable_risk")

    def test_configuration_requires_server_side_provider(self):
        with self.assertRaisesRegex(ValueError, "server-controlled"):
            TestbedState(TokenRegistry({"operator-B": secrets.token_urlsafe(24)}),
                         risk_mode="risk_gated")

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            decide(True, mode="risk_gated", risk_score=0.4, threshold=1.0)


class RiskGateHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sensor_token = secrets.token_urlsafe(32)
        cls.operator_token = secrets.token_urlsafe(32)
        # No HTTP header can modify this mapping; it represents server-side evidence.
        cls.risk_evidence = {"sensor-A": 0.1, "operator-B": 0.1}

        def provider(principal, method, path):
            value = cls.risk_evidence.get(principal)
            if isinstance(value, Exception):
                raise value
            return value

        cls.state = TestbedState(
            TokenRegistry({"sensor-A": cls.sensor_token, "operator-B": cls.operator_token}),
            risk_mode="risk_gated", risk_provider=provider, risk_threshold=0.7,
        )
        cls.server = make_server(cls.state, port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.risk_evidence["sensor-A"] = 0.1
        self.risk_evidence["operator-B"] = 0.1
        with self.state.lock:
            self.state.sampling_interval = 30
            self.state.audit.clear()

    def request(self, method, path, token=None, payload=None, extra_headers=None):
        headers = dict(extra_headers or {})
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        body = json.dumps(payload).encode() if payload is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = Request(self.base + path, data=body, method=method, headers=headers)
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_authorized_operator_low_risk_updates_configuration(self):
        status, response = self.request("POST", "/v1/config", self.operator_token,
                                        {"sampling_interval": 11})
        self.assertEqual(status, 200)
        self.assertEqual(response["sampling_interval"], 11)
        self.assertEqual(self.state.sampling_interval, 11)

    def test_authorized_operator_high_risk_write_blocked_without_mutation(self):
        self.risk_evidence["operator-B"] = 0.9
        status, response = self.request("POST", "/v1/config", self.operator_token,
                                        {"sampling_interval": 11})
        self.assertEqual(status, 403)
        self.assertEqual(response["error"], "denied_by_risk")
        self.assertEqual(self.state.sampling_interval, 30)
        self.assertEqual(self.state.audit[-1]["decision"], "denied_by_risk")

    def test_client_spoofed_low_risk_cannot_override_high_server_risk(self):
        self.risk_evidence["operator-B"] = 0.99
        status, _ = self.request("POST", "/v1/config", self.operator_token,
                                 {"sampling_interval": 11},
                                 {"X-Risk": "0.0", "X-Principal": "operator-B"})
        self.assertEqual(status, 403)
        self.assertEqual(self.state.sampling_interval, 30)

    def test_sensor_low_risk_never_gets_write_permission(self):
        status, response = self.request("POST", "/v1/config", self.sensor_token,
                                        {"sampling_interval": 11}, {"X-Risk": "0.0"})
        self.assertEqual(status, 403)
        self.assertEqual(response["error"], "permission_denied")
        self.assertEqual(self.state.sampling_interval, 30)

    def test_missing_risk_evidence_returns_503_and_prevents_write(self):
        self.risk_evidence["operator-B"] = None
        status, response = self.request("POST", "/v1/config", self.operator_token,
                                        {"sampling_interval": 11})
        self.assertEqual(status, 503)
        self.assertEqual(response["error"], "unavailable_risk")
        self.assertEqual(self.state.sampling_interval, 30)

    def test_risk_provider_exception_does_not_allow_write(self):
        self.risk_evidence["operator-B"] = RuntimeError("model offline")
        status, _ = self.request("POST", "/v1/config", self.operator_token,
                                 {"sampling_interval": 11})
        self.assertEqual(status, 503)
        self.assertEqual(self.state.sampling_interval, 30)

    def test_valid_authentication_required_before_risk_check(self):
        status, _ = self.request("POST", "/v1/config", None, {"sampling_interval": 11})
        self.assertEqual(status, 401)
        self.assertEqual(self.state.sampling_interval, 30)


if __name__ == "__main__":
    unittest.main()
