from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> None:
    root = Path("cloud/keys")
    root.mkdir(parents=True, exist_ok=True)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    (root / "demo_ed25519_private.pem").write_bytes(private_pem)

    public_raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    (root / "demo_ed25519_public_hex.txt").write_text(public_raw.hex(), encoding="utf-8")

    print("Generated cloud/keys/demo_ed25519_private.pem")
    print("Generated cloud/keys/demo_ed25519_public_hex.txt")


if __name__ == "__main__":
    main()
