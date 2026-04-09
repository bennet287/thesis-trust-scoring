use anyhow::{anyhow, Context, Result};
use flate2::read::GzDecoder;
use serde_json::json;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tar::Archive;

fn ensure_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {:?}", path))?;
    Ok(())
}

fn copy_if_exists(src: &Path, dst: &Path) -> Result<()> {
    if src.exists() {
        fs::copy(src, dst).with_context(|| format!("failed to copy {:?} -> {:?}", src, dst))?;
    }
    Ok(())
}

pub fn apply_update_atomically(package_bytes: &[u8], version: &str, work_dir: &str) -> Result<()> {
    let root = PathBuf::from(work_dir);
    let active_dir = root.join("active");
    let backup_dir = root.join("backup");
    let staging_parent = root.join("staging");
    let versions_dir = root.join("versions");
    let reload_trigger = root.join(".reload_trigger");
    let version_file = root.join("current_version.json");

    ensure_dir(&active_dir)?;
    ensure_dir(&backup_dir)?;
    ensure_dir(&staging_parent)?;
    ensure_dir(&versions_dir)?;

    let tmp = tempdir_in(&staging_parent)?;
    let extract_dir = tmp.path().join("extract");
    ensure_dir(&extract_dir)?;

    unpack_tar_gz(package_bytes, &extract_dir)?;

    let model = extract_dir.join("model.json");
    let kb = extract_dir.join("kb.json");
    let policy = extract_dir.join("policy.json");

    if !model.exists() || !kb.exists() || !policy.exists() {
        return Err(anyhow!("package missing required files"));
    }

    let candidate_version_dir = versions_dir.join(version);
    ensure_dir(&candidate_version_dir)?;
    fs::copy(&model, candidate_version_dir.join("model.json"))?;
    fs::copy(&kb, candidate_version_dir.join("kb.json"))?;
    fs::copy(&policy, candidate_version_dir.join("policy.json"))?;

    // Backup current active set
    copy_if_exists(&active_dir.join("model.json"), &backup_dir.join("model.json"))?;
    copy_if_exists(&active_dir.join("kb.json"), &backup_dir.join("kb.json"))?;
    copy_if_exists(&active_dir.join("policy.json"), &backup_dir.join("policy.json"))?;

    // Atomic-ish replacement by writing temp then rename
    replace_file_atomic(&model, &active_dir.join("model.json"))?;
    replace_file_atomic(&kb, &active_dir.join("kb.json"))?;
    replace_file_atomic(&policy, &active_dir.join("policy.json"))?;

    fs::write(
        &version_file,
        serde_json::to_vec_pretty(&json!({ "version": version }))?,
    )
    .context("failed to write current_version.json")?;

    fs::write(&reload_trigger, b"reload").context("failed to write reload trigger")?;

    Ok(())
}

pub fn rollback(work_dir: &str) -> Result<()> {
    let root = PathBuf::from(work_dir);
    let active_dir = root.join("active");
    let backup_dir = root.join("backup");

    for name in ["model.json", "kb.json", "policy.json"] {
        let src = backup_dir.join(name);
        let dst = active_dir.join(name);
        if src.exists() {
            replace_file_atomic(&src, &dst)?;
        }
    }

    Ok(())
}

pub fn load_local_version(work_dir: &str) -> String {
    let version_file = PathBuf::from(work_dir).join("current_version.json");
    let raw = fs::read_to_string(version_file);
    if let Ok(s) = raw {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(version) = v.get("version").and_then(|x| x.as_str()) {
                return version.to_string();
            }
        }
    }
    "0.0.0".to_string()
}

fn unpack_tar_gz(bytes: &[u8], out_dir: &Path) -> Result<()> {
    let cursor = Cursor::new(bytes);
    let gz = GzDecoder::new(cursor);
    let mut archive = Archive::new(gz);
    archive.unpack(out_dir).context("failed to unpack update archive")?;
    Ok(())
}

fn replace_file_atomic(src: &Path, dst: &Path) -> Result<()> {
    let tmp = dst.with_extension("tmp");
    fs::copy(src, &tmp).with_context(|| format!("failed to stage {:?}", dst))?;
    fs::rename(&tmp, dst).with_context(|| format!("failed to rename into {:?}", dst))?;
    Ok(())
}

fn tempdir_in(base: &Path) -> Result<tempfile::TempDir> {
    tempfile::Builder::new()
        .prefix("update_")
        .tempdir_in(base)
        .context("failed to create temp dir")
}
