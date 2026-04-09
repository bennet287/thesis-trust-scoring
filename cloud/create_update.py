import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


REQUIRED_FILES = ["model.json", "kb.json", "policy.json"]


def _load_private_key(key_file: Path, password: Optional[str]) -> Ed25519PrivateKey:
    raw = key_file.read_bytes()
    pwd = password.encode("utf-8") if password else None
    key = serialization.load_pem_private_key(raw, password=pwd)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key must be Ed25519")
    return key


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_package(input_dir: Path, output_package: Path) -> None:
    output_package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_package, mode="w:gz") as tf:
        for name in REQUIRED_FILES:
            file_path = input_dir / name
            if not file_path.exists():
                raise FileNotFoundError(f"required file missing: {file_path}")
            tf.add(file_path, arcname=name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create signed LTCA update package + manifest")
    parser.add_argument("--version", required=True, help="Artifact version, e.g. 1.1.0")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing model.json, kb.json, policy.json",
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
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    package_path = out_dir / f"update_{version}.tar.gz"
    manifest_path = out_dir / f"manifest_{version}.json"

    _build_package(input_dir, package_path)

    package_bytes = package_path.read_bytes()
    key = _load_private_key(Path(args.private_key), args.password)
    signature_hex = key.sign(package_bytes).hex()

    manifest = {
        "version": version,
        "hash": _sha256_hex(package_bytes),
        "size": len(package_bytes),
        "signature": signature_hex,
        "files": REQUIRED_FILES,
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

    print(json.dumps({
        "package": str(package_path),
        "manifest": str(manifest_path),
        "manifest_content": manifest,
        "check_response_example": check_response_example,
    }, indent=2))


if __name__ == "__main__":
    main()
