# LTCA User Manual (Panel Demo)

This manual explains how to run the LTCA system from a fresh setup to a complete end-to-end demonstration, including where outputs, metrics, and evidence are stored.

## 1. Scope and Outcome

After completing this guide, you will have demonstrated:

- secure cloud-to-edge artifact synchronization
- integrity verification (size, hash, signature)
- atomic activation of synchronized artifacts
- runtime reload of KB/policy at the host
- policy-driven trust/action behavior

Target validated version in this project: **v1.2.0**.

---

## 2. End-to-End Architecture

LTCA consists of four cooperating layers:

1. Inference layer (Wasm model)
- deterministic lightweight edge model
- compiled in `wasm_module/`

2. Runtime orchestration layer (host)
- loads model outputs + retrieval KB + policy
- computes fused trust and final action
- supports trigger-based reload from synchronized state
- implemented in `host_runner/`

3. Synchronization layer (edge sync client)
- polls control plane for updates
- downloads manifest + package
- verifies size/hash/signature (Ed25519)
- performs atomic apply and version update
- implemented in `edge/sync_client/`

4. Control plane layer (cloud demo scripts)
- builds signed update package and manifest
- serves check/manifest/download endpoints
- implemented in `cloud/`

### Data flow

1. Cloud builds `update_<version>.tar.gz` + `manifest_<version>.json`.
2. Sync client receives update metadata from `/check/<device_id>`.
3. Sync client verifies package integrity and signature.
4. Sync client activates files into `sync_state/active` and writes version.
5. Host runner consumes active KB/policy and applies runtime behavior.

---

## 3. Repository Map (What matters for panel demo)

- `cloud/create_update.py`: build package + manifest
- `cloud/sign_package.py`: signing utility
- `cloud/mock_control_plane.py`: local control-plane server
- `cloud/gen_demo_keys.py`: demo Ed25519 key generation
- `cloud/updates/`: generated manifests and update archives
- `cloud/demo_input/1.1.0/`: payload template (`model.json`, `kb.json`, `policy.json`)

- `edge/sync_client/src/main.rs`: sync loop + evidence logs
- `edge/sync_client/src/verify.rs`: hash/signature verification

- `host_runner/src/main.rs`: trust fusion, action policy, reload trigger handling

- `wasm_module/`: Wasm model crate

- `sync_state/`: runtime synchronized state (`active`, `backup`, `versions`, `current_version.json`)

- `results/models/policy.json`: final tuned runtime policy
- `results/metrics/`: calibration outputs
- `THESIS_DEMO_REPRO.md`: compact reproducibility checklist

---

## 4. Prerequisites (Fresh Machine)

## 4.1 Tools

Install:

- Git
- Rust (stable)
- Python 3.11
- Conda (optional but recommended)

## 4.2 Rust target for Wasm

```powershell
rustup target add wasm32-unknown-unknown
```

## 4.3 Python environment

From repository root:

```powershell
conda env create -f environment.yml
conda activate thesis
pip install cryptography
```

`cryptography` is required for Ed25519 key handling/signing.

---

## 5. One-Time Build

From repository root:

```powershell
# Build Wasm inference module
Set-Location .\wasm_module
cargo build --release --target wasm32-unknown-unknown

# Build host runner
Set-Location ..\host_runner
cargo build

# Build sync client
Set-Location ..\edge\sync_client
cargo build

# Return to repo root
Set-Location ..\..\
```

---

## 6. Demo Run (From Scratch, End to End)

Use separate terminals as indicated.

### Terminal A: Repository root setup and clean state

```powershell
Set-Location D:\thesis-trust-scoring

# Clean synchronization state
Remove-Item .\sync_state\* -Recurse -Force -ErrorAction SilentlyContinue
mkdir .\sync_state\active, .\sync_state\backup, .\sync_state\staging, .\sync_state\versions -Force | Out-Null
```

### Terminal A: Generate demo keys (if not already present)

```powershell
python .\cloud\gen_demo_keys.py
```

Expected files:

- `cloud/keys/demo_ed25519_private.pem` (local only; ignored by git)
- `cloud/keys/demo_ed25519_public_hex.txt`

### Terminal A: Create signed update package (v1.2.0)

```powershell
python .\cloud\create_update.py `
  --version 1.2.0 `
  --input-dir .\cloud\demo_input\1.1.0 `
  --private-key .\cloud\keys\demo_ed25519_private.pem `
  --base-url http://127.0.0.1:5001 `
  --out-dir .\cloud\updates
```

Expected outputs:

- `cloud/updates/update_1.2.0.tar.gz`
- `cloud/updates/manifest_1.2.0.json`

### Terminal B: Start local control plane

```powershell
Set-Location D:\thesis-trust-scoring
python .\cloud\mock_control_plane.py `
  --manifest .\cloud\updates\manifest_1.2.0.json `
  --package .\cloud\updates\update_1.2.0.tar.gz `
  --host 127.0.0.1 `
  --port 5001
```

Expected startup lines:

- `Serving mock control plane at http://127.0.0.1:5001`
- `Version: 1.2.0`

### Terminal A: Endpoint smoke test

```powershell
Invoke-RestMethod http://127.0.0.1:5001/check/edge-001
Invoke-RestMethod http://127.0.0.1:5001/manifest/1.2.0
```

Optional download check:

```powershell
Invoke-WebRequest http://127.0.0.1:5001/download/1.2.0 -OutFile .\cloud\updates\_download_check_1.2.0.tar.gz
```

### Terminal C: Run sync client

```powershell
Set-Location D:\thesis-trust-scoring\edge\sync_client
$env:SYNC_BASE_URL = "http://127.0.0.1:5001"
$env:SYNC_DEVICE_ID = "edge-001"
$env:SYNC_POLL_SECONDS = "5"
$env:SYNC_PUBLIC_KEY_HEX = (Get-Content ..\..\cloud\keys\demo_ed25519_public_hex.txt -Raw).Trim()
$env:SYNC_WORK_DIR = "..\..\sync_state"
$env:RUST_LOG = "info"
cargo run
```

Expected evidence lines include:

- `current local version: ...`
- `downloading manifest for version 1.2.0`
- `downloading package for version 1.2.0`
- `package size verified`
- `package hash verified`
- `package signature verified`
- `update 1.2.0 applied successfully`
- repeated same-version polls: `ignoring non-monotonic version...`

### Terminal A: Verify synchronized filesystem state

```powershell
Set-Location D:\thesis-trust-scoring
Get-Content .\sync_state\current_version.json
Get-ChildItem .\sync_state\active
```

Expected:

- version is `1.2.0`
- `sync_state/active` contains:
  - `model.json`
  - `kb.json`
  - `policy.json`

### Terminal D: Run host runner and show runtime behavior

```powershell
Set-Location D:\thesis-trust-scoring\host_runner
cargo run
```

Expected behavior trace:

- trust/action per step (t1..t5)
- final action sequence under tuned policy

---

## 7. Explicit Reload Evidence Procedure (Panel-Friendly)

Use this when you want to prove reload happened during execution (not just startup load).

### Step 1: Prepare baseline and clear trigger

```powershell
Set-Location D:\thesis-trust-scoring
Remove-Item .\sync_state\.reload_trigger -Force -ErrorAction SilentlyContinue

Move-Item .\sync_state\active\policy.json .\sync_state\active\policy.sync.bak -Force
Move-Item .\sync_state\active\kb.json .\sync_state\active\kb.sync.bak -Force
if (Test-Path .\sync_state\active\model.json) {
  Move-Item .\sync_state\active\model.json .\sync_state\active\model.sync.bak -Force
}
```

### Step 2: Start host with demo pacing (Terminal D)

```powershell
Set-Location D:\thesis-trust-scoring\host_runner
$env:DEMO_STEP_SLEEP_MS = "300"
cargo run
```

### Step 3: Restore synced files and trigger reload while host is running (Terminal A)

```powershell
Set-Location D:\thesis-trust-scoring
Move-Item .\sync_state\active\policy.sync.bak .\sync_state\active\policy.json -Force
Move-Item .\sync_state\active\kb.sync.bak .\sync_state\active\kb.json -Force
if (Test-Path .\sync_state\active\model.sync.bak) {
  Move-Item .\sync_state\active\model.sync.bak .\sync_state\active\model.json -Force
}
Set-Content .\sync_state\.reload_trigger "reload"
```

Expected host evidence line:

- `Reloaded KB and policy from sync state`

---

## 8. Evidence Checklist for Panel

Capture these three evidence groups:

1. Sync proof
- successful apply + verification logs from sync client

2. Reload proof
- explicit host line: `Reloaded KB and policy from sync state`

3. Behavior proof
- same sample / sequence producing policy-governed action behavior

---

## 9. Where Results and Evidence Are Saved

### Runtime synchronization evidence

- `sync_state/current_version.json`
- `sync_state/active/model.json`
- `sync_state/active/kb.json`
- `sync_state/active/policy.json`
- `sync_state/versions/<version>/...`

### Cloud update evidence

- `cloud/updates/manifest_1.2.0.json`
- `cloud/updates/update_1.2.0.tar.gz`

### Calibration evidence

- `results/metrics/policy_calibration_weights_best_dt.json`
- `results/metrics/policy_calibration_thresholds_best_dt.json`

### Final runtime policy

- `results/models/policy.json`

### Reproducibility documentation

- `THESIS_DEMO_REPRO.md`
- `LTCA_PANEL_USER_MANUAL.md` (this file)

Note: runtime logs and generated `.jsonl` traces under `results/logs/` are ignored for git hygiene.

---

## 10. Troubleshooting

## 10.1 Port conflict on control plane

If `5000`/`5001` is busy, choose another port and update:

- `--base-url` in `create_update.py`
- `--port` in `mock_control_plane.py`
- `SYNC_BASE_URL` for sync client

## 10.2 Signature verification failure

Check that:

- package was signed with the matching private key
- `SYNC_PUBLIC_KEY_HEX` matches `cloud/keys/demo_ed25519_public_hex.txt`

## 10.3 Host reload line not visible

- ensure trigger file is written while host is still running
- set `DEMO_STEP_SLEEP_MS=300` to widen timing window
- confirm host checks path `../sync_state/.reload_trigger`

## 10.4 Wasm artifact missing

Rebuild Wasm:

```powershell
Set-Location D:\thesis-trust-scoring\wasm_module
cargo build --release --target wasm32-unknown-unknown
```

---

## 11. Panel Demo Script (Short Narrative)

Use this live narrative:

1. "I generate a signed update package and manifest for v1.2.0."
2. "The sync client polls control plane and verifies size/hash/signature."
3. "The update is atomically activated in sync_state and version is advanced."
4. "The host reloads synchronized KB/policy via trigger without modifying the Wasm model."
5. "Trust-action behavior reflects calibrated policy governance under the same runtime stack."

---

## 12. Final System Status Statement

LTCA v1.2.0 is implementation-complete, reproducible, Git-clean, and validated end to end with secure artifact synchronization, runtime KB/policy reload, and calibrated policy behavior.
