use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct CheckResponse {
    pub available: bool,
    pub version: Option<String>,
    pub hash: Option<String>,
    pub size: Option<u64>,
    pub manifest_url: Option<String>,
    pub package_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpdateManifest {
    pub version: String,
    pub hash: String,
    pub size: u64,
    pub signature: String, // hex-encoded Ed25519 signature over package bytes
    pub files: Vec<String>,
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalVersion {
    pub version: String,
}

#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub base_url: String,
    pub device_id: String,
    pub poll_seconds: u64,
    pub public_key_hex: String,
    pub work_dir: String,
}
