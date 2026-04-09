use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::types::UpdateManifest;

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

pub fn verify_hash(bytes: &[u8], expected_hex: &str) -> Result<()> {
    let actual = sha256_hex(bytes);
    if actual != expected_hex {
        return Err(anyhow!(
            "hash mismatch: expected {}, got {}",
            expected_hex,
            actual
        ));
    }
    Ok(())
}

pub fn verify_size(bytes: &[u8], expected: u64) -> Result<()> {
    let actual = bytes.len() as u64;
    if actual != expected {
        return Err(anyhow!(
            "size mismatch: expected {}, got {}",
            expected,
            actual
        ));
    }
    Ok(())
}

pub fn verify_signature(bytes: &[u8], manifest: &UpdateManifest, public_key_hex: &str) -> Result<()> {
    let pk_bytes = hex::decode(public_key_hex).context("invalid public key hex")?;
    let pk_array: [u8; 32] = pk_bytes
        .try_into()
        .map_err(|_| anyhow!("public key must be 32 bytes"))?;
    let verifying_key = VerifyingKey::from_bytes(&pk_array).context("invalid Ed25519 public key")?;

    let sig_bytes = hex::decode(&manifest.signature).context("invalid signature hex")?;
    let sig_array: [u8; 64] = sig_bytes
        .try_into()
        .map_err(|_| anyhow!("signature must be 64 bytes"))?;
    let signature = Signature::from_bytes(&sig_array);

    verifying_key
        .verify(bytes, &signature)
        .context("signature verification failed")?;

    Ok(())
}
