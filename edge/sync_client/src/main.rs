mod api;
mod swap;
mod types;
mod verify;

use anyhow::{Context, Result};
use log::{error, info, warn};
use reqwest::Client;
use std::env;
use std::time::Duration;
use tokio::time::sleep;

use crate::api::{check_for_update, download_manifest, download_package_bytes};
use crate::swap::{apply_update_atomically, load_local_version, rollback};
use crate::types::ClientConfig;
use crate::verify::{verify_hash, verify_signature, verify_size};

fn read_config() -> ClientConfig {
    ClientConfig {
        base_url: env::var("SYNC_BASE_URL").unwrap_or_else(|_| "https://127.0.0.1:5000".into()),
        device_id: env::var("SYNC_DEVICE_ID").unwrap_or_else(|_| "edge-001".into()),
        poll_seconds: env::var("SYNC_POLL_SECONDS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(60),
        public_key_hex: env::var("SYNC_PUBLIC_KEY_HEX").unwrap_or_default(),
        work_dir: env::var("SYNC_WORK_DIR").unwrap_or_else(|_| "../sync_state".into()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cfg = read_config();

    if cfg.public_key_hex.is_empty() {
        anyhow::bail!("SYNC_PUBLIC_KEY_HEX is required");
    }

    let client = Client::builder()
        .danger_accept_invalid_certs(true) // thesis demo only; remove later
        .build()
        .context("failed to build HTTP client")?;

    loop {
        if let Err(e) = run_once(&client, &cfg).await {
            error!("sync loop error: {e:#}");
        }
        sleep(Duration::from_secs(cfg.poll_seconds)).await;
    }
}

async fn run_once(client: &Client, cfg: &ClientConfig) -> Result<()> {
    let current_version = load_local_version(&cfg.work_dir);
    info!("current local version: {}", current_version);

    let check = check_for_update(client, &cfg.base_url, &cfg.device_id).await?;
    if !check.available {
        info!("no update available");
        return Ok(());
    }

    let new_version = check
        .version
        .clone()
        .context("server says update available but version missing")?;

    if new_version <= current_version {
        warn!(
            "ignoring non-monotonic version: current={}, incoming={}",
            current_version, new_version
        );
        return Ok(());
    }

    let manifest_url = check
        .manifest_url
        .clone()
        .unwrap_or_else(|| format!("{}/manifest/{}", cfg.base_url, new_version));

    let package_url = check
        .package_url
        .clone()
        .unwrap_or_else(|| format!("{}/download/{}", cfg.base_url, new_version));

    info!("downloading manifest for version {}", new_version);
    let manifest = download_manifest(client, &manifest_url).await?;

    info!("downloading package for version {}", manifest.version);
    let package_bytes = download_package_bytes(client, &package_url).await?;

    verify_size(&package_bytes, manifest.size)?;
    info!("package size verified");
    verify_hash(&package_bytes, &manifest.hash)?;
    info!("package hash verified");
    verify_signature(&package_bytes, &manifest, &cfg.public_key_hex)?;
    info!("package signature verified");

    match apply_update_atomically(&package_bytes, &manifest.version, &cfg.work_dir) {
        Ok(_) => {
            info!("update {} applied successfully", manifest.version);
            Ok(())
        }
        Err(e) => {
            error!("apply failed: {e:#}; attempting rollback");
            rollback(&cfg.work_dir)?;
            Err(e)
        }
    }
}
