import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _load_private_key(key_file: Path, password: Optional[str]) -> Ed25519PrivateKey:
    raw = key_file.read_bytes()
    pwd = password.encode("utf-8") if password else None
    key = serialization.load_pem_private_key(raw, password=pwd)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key must be Ed25519")
    return key


def sign_bytes(private_key: Ed25519PrivateKey, data: bytes) -> str:
    signature = private_key.sign(data)
    return signature.hex()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign an update package with Ed25519")
    parser.add_argument("--package", required=True, help="Path to .tar.gz package")
    parser.add_argument("--private-key", required=True, help="Path to Ed25519 PEM private key")
    parser.add_argument("--password", default=None, help="Optional PEM password")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON containing hash,size,signature",
    )

    args = parser.parse_args()

    package_path = Path(args.package)
    key_path = Path(args.private_key)

    data = package_path.read_bytes()
    key = _load_private_key(key_path, args.password)

    payload = {
        "hash": sha256_hex(data),
        "size": len(data),
        "signature": sign_bytes(key, data),
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
