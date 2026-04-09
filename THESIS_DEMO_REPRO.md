# LTCA v1.1 Reproducible Demo

This document reproduces the secure cloud-assisted update prototype end to end.

## Prerequisites

- Python environment with `cryptography` installed.
- Rust toolchain installed.
- Built host runner and sync client dependencies at least once.

## 1) Generate demo signing keys

Run from repository root:

```powershell
python cloud/gen_demo_keys.py
```

Expected files:

- `cloud/keys/demo_ed25519_private.pem`
- `cloud/keys/demo_ed25519_public_hex.txt`

## 2) Build signed update artifacts

Use existing demo input payload and create signed versioned package and manifest.

```powershell
python cloud/create_update.py --version 1.1.1 --input-dir cloud/demo_input/1.1.0 --private-key cloud/keys/demo_ed25519_private.pem --out-dir cloud/updates --base-url http://127.0.0.1:5000
```

Expected files:

- `cloud/updates/update_1.1.1.tar.gz`
- `cloud/updates/manifest_1.1.1.json`

## 3) Start local mock control plane

```powershell
python cloud/mock_control_plane.py --manifest cloud/updates/manifest_1.1.1.json --package cloud/updates/update_1.1.1.tar.gz --host 127.0.0.1 --port 5000
```

Expected log includes:

- `Serving mock control plane at http://127.0.0.1:5000`
- `Version: 1.1.1`

## 4) Run Rust sync client against control plane

Open a second terminal from `edge/sync_client` and set environment variables:

```powershell
$env:SYNC_BASE_URL="http://127.0.0.1:5000"
$env:SYNC_DEVICE_ID="edge-001"
$env:SYNC_POLL_SECONDS="5"
$env:SYNC_PUBLIC_KEY_HEX=(Get-Content ..\..\cloud\keys\demo_ed25519_public_hex.txt -Raw).Trim()
$env:SYNC_WORK_DIR="..\..\sync_state"
$env:RUST_LOG="info"
cargo run
```

Expected sync client log sequence:

- `current local version: ...`
- `downloading manifest for version 1.1.1`
- `downloading package for version 1.1.1`
- `update 1.1.1 applied successfully`

Then repeated checks should reject same-version non-monotonic updates.

## 5) Verify active sync state

```powershell
Get-Content sync_state/current_version.json
Get-ChildItem sync_state/active
```

Expected:

- `current_version.json` shows `1.1.1`
- `sync_state/active` contains `model.json`, `kb.json`, `policy.json`

## 6) Run host runner and observe reload

From `host_runner`:

```powershell
cargo run
```

Expected log line:

- `Reloaded KB and policy from sync state`

## 7) Capture policy-effect evidence

For a fixed sample where `T = 0.958322` was observed:

- baseline policy action: `ALLOW`
- synced policy action: `LIMIT`

This demonstrates behavior change through synchronized policy update without changing the Wasm inference module.

## 8) Verification overhead reference

Observed one-shot verification numbers from local run:

- package size: `6634` bytes
- SHA-256: `0.058 ms`
- Ed25519 verify: `0.428 ms`

## 9) Optional: tune policy for results chapter

Tune fusion weights:

```powershell
python python_baseline/tune_weights.py --model dt --apply
```

Tune thresholds:

```powershell
python python_baseline/tune_thresholds.py --model dt --apply
```

Produced artifacts:

- `results/metrics/policy_calibration_weights_best_dt.json`
- `results/metrics/policy_calibration_thresholds_best_dt.json`
