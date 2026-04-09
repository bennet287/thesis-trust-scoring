use anyhow::{Context, Result};
use reqwest::Client;

use crate::types::{CheckResponse, UpdateManifest};

pub async fn check_for_update(client: &Client, base_url: &str, device_id: &str) -> Result<CheckResponse> {
    let url = format!("{}/check/{}", base_url.trim_end_matches('/'), device_id);
    let res = client
        .get(url)
        .send()
        .await
        .context("failed to send /check request")?
        .error_for_status()
        .context("non-success /check response")?
        .json::<CheckResponse>()
        .await
        .context("failed to parse /check response")?;
    Ok(res)
}

pub async fn download_manifest(client: &Client, manifest_url: &str) -> Result<UpdateManifest> {
    let manifest = client
        .get(manifest_url)
        .send()
        .await
        .context("failed to download manifest")?
        .error_for_status()
        .context("non-success manifest response")?
        .json::<UpdateManifest>()
        .await
        .context("failed to parse manifest")?;
    Ok(manifest)
}

pub async fn download_package_bytes(client: &Client, package_url: &str) -> Result<Vec<u8>> {
    let bytes = client
        .get(package_url)
        .send()
        .await
        .context("failed to download package")?
        .error_for_status()
        .context("non-success package response")?
        .bytes()
        .await
        .context("failed to read package bytes")?;
    Ok(bytes.to_vec())
}
