import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Optional

import cbor2
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _load_private_key(key_file: Path, password: Optional[str]) -> Ed25519PrivateKey:
    raw = key_file.read_bytes()
    pwd = password.encode("utf-8") if password else None
    key = serialization.load_pem_private_key(raw, password=pwd)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key must be Ed25519")
    return key


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ensure_compact_kb(input_dir: Path) -> Path:
    kb_cbor_path = input_dir / "kb.cbor"
    if kb_cbor_path.exists():
        return kb_cbor_path

    retrieval_cbor = input_dir / "retrieval_kb.cbor"
    if retrieval_cbor.exists():
        return retrieval_cbor

    kb_json_path = input_dir / "kb.json"
    retrieval_json = input_dir / "retrieval_kb.json"

    src_json = kb_json_path if kb_json_path.exists() else retrieval_json
    if not src_json.exists():
        raise FileNotFoundError(
            f"required compact KB input missing: {kb_json_path} or {retrieval_json}"
        )

    kb = json.loads(src_json.read_text(encoding="utf-8"))
    target = input_dir / "retrieval_kb.cbor"
    target.write_bytes(cbor2.dumps(kb))
    return target


def _build_package(input_dir: Path, output_package: Path, profile: str) -> list[str]:
    output_package.parent.mkdir(parents=True, exist_ok=True)

    model_path = input_dir / "model.json"
    policy_path = input_dir / "policy.json"
    if not model_path.exists():
        raise FileNotFoundError(f"required file missing: {model_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"required file missing: {policy_path}")

    if profile == "compact":
        kb_path = _ensure_compact_kb(input_dir)
        kb_arcname = "kb.cbor"
    else:
        kb_json_path = input_dir / "kb.json"
        retrieval_json = input_dir / "retrieval_kb.json"
        kb_path = kb_json_path if kb_json_path.exists() else retrieval_json
        if not kb_path.exists():
            raise FileNotFoundError(
                f"required full KB input missing: {kb_json_path} or {retrieval_json}"
            )
        kb_arcname = "kb.json"

    files = ["model.json", kb_arcname, "policy.json"]

    with tarfile.open(output_package, mode="w:gz") as tf:
        tf.add(model_path, arcname="model.json")
        tf.add(kb_path, arcname=kb_arcname)
        tf.add(policy_path, arcname="policy.json")

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Create signed LTCA update package + manifest")
    parser.add_argument("--version", required=True, help="Artifact version, e.g. 1.2.1")
    parser.add_argument(
        "--profile",
        choices=["full", "compact"],
        default="full",
        help="Deployment profile",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing model/policy and kb.json or retrieval_kb.json",
    )
    parser.add_argument(
        "--private-key",
        required=True,
        help="Path to Ed25519 PEM private key",
    )
    parser.add_argument("--password", default=None, help="Optional PEM password")
    parser.add_argument(
        "--base-url",
        default="https://127.0.0.1:5000",
        help="Base URL used in generated manifest/check payload snippets",
    )
    parser.add_argument(
        "--out-dir",
        default="updates",
        help="Output directory for package and manifest",
    )

    args = parser.parse_args()

    version = args.version
    profile = args.profile
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    package_name = f"update_{version}.tar.gz" if profile == "full" else f"update_{version}_{profile}.tar.gz"
    package_path = out_dir / package_name
    manifest_name = f"manifest_{version}.json" if profile == "full" else f"manifest_{version}_{profile}.json"
    manifest_path = out_dir / manifest_name

    packaged_files = _build_package(input_dir, package_path, profile)

    package_bytes = package_path.read_bytes()
    key = _load_private_key(Path(args.private_key), args.password)
    signature_hex = key.sign(package_bytes).hex()

    manifest = {
        "version": version,
        "profile": profile,
        "hash": _sha256_hex(package_bytes),
        "size": len(package_bytes),
        "signature": signature_hex,
        "files": packaged_files,
        "created_at": None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    check_response_example = {
        "available": True,
        "version": version,
        "hash": manifest["hash"],
        "size": manifest["size"],
        "manifest_url": f"{args.base_url.rstrip('/')}/manifest/{version}",
        "package_url": f"{args.base_url.rstrip('/')}/download/{version}",
    }

    print(
        json.dumps(
            {
                "package": str(package_path),
                "manifest": str(manifest_path),
                "manifest_content": manifest,
                "check_response_example": check_response_example,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()