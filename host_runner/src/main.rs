use anyhow::{anyhow, Result};
use std::env;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;
use wasmtime::{Engine, Instance, Module, Store};

fn action_name(code: i32) -> &'static str {
    match code {
        0 => "ALLOW",
        1 => "LIMIT",
        2 => "QUARANTINE",
        3 => "BLOCK",
        _ => "UNKNOWN",
    }
}

#[derive(Debug, serde::Deserialize)]
struct SequenceSample {
    values: Vec<f64>,
    attack_probability: f64,
    trust: f64,
}

#[derive(Debug, serde::Deserialize)]
struct RetrievalKB {
    feature_names: Vec<String>,
    scaler_min: Vec<f64>,
    scaler_max: Vec<f64>,
    profiles: Vec<RetrievalProfile>,
}

#[derive(Debug, serde::Deserialize)]
struct RetrievalProfile {
    name: String,
    description: String,
    profile_trust: f64,
    count: usize,
    centroid: Vec<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct FusionBand {
    w_m: f64,
    w_r: f64,
    w_h: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct FusionConfig {
    high_confidence: FusionBand,
    medium_confidence: FusionBand,
    low_confidence: FusionBand,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ConfidenceGates {
    high: f64,
    medium: f64,
    low_center_min: f64,
    high_center_max: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Thresholds {
    allow: f64,
    limit: f64,
    quarantine: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PolicyConfig {
    fusion: FusionConfig,
    confidence_gates: ConfidenceGates,
    thresholds: Thresholds,
}

fn compute_history_score(history: &[f64], current: f64) -> f64 {
    if history.is_empty() {
        return current;
    }

    let avg = history.iter().sum::<f64>() / history.len() as f64;
    let deviation = (current - avg).abs();
    (1.0 - deviation).clamp(0.0, 1.0)
}

fn normalize_with_minmax(values: &[f64], mins: &[f64], maxs: &[f64]) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let min = mins[i];
            let max = maxs[i];
            if (max - min).abs() < 1e-12 {
                0.0
            } else {
                (x - min) / (max - min)
            }
        })
        .collect()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn compute_retrieval_score(values: &[f64], kb: &RetrievalKB) -> (f64, String, f64, f64) {
    let x_norm = normalize_with_minmax(values, &kb.scaler_min, &kb.scaler_max);

    let mut best_profile: Option<&RetrievalProfile> = None;
    let mut best_distance = f64::INFINITY;

    for profile in &kb.profiles {
        let dist = euclidean_distance(&x_norm, &profile.centroid);
        if dist < best_distance {
            best_distance = dist;
            best_profile = Some(profile);
        }
    }

    let profile = best_profile.expect("Retrieval KB has no profiles");

    let max_dist = (x_norm.len() as f64).sqrt();
    let similarity = (1.0 - best_distance / max_dist).clamp(0.0, 1.0);

    // Retrieval trust is interpolated between the profile trust and a neutral 0.5,
    // depending on how similar the sample is to the matched profile.
    let retrieval_trust = similarity * profile.profile_trust + (1.0 - similarity) * 0.5;

    (
        retrieval_trust,
        profile.name.clone(),
        similarity,
        profile.profile_trust,
    )
}

fn load_kb(path: &str) -> Result<RetrievalKB> {
    let kb_str = fs::read_to_string(path)?;
    let kb: RetrievalKB = serde_json::from_str(&kb_str)?;
    Ok(kb)
}

fn load_policy(path: &str) -> Result<PolicyConfig> {
    let raw = fs::read_to_string(path)?;
    let policy: PolicyConfig = serde_json::from_str(&raw)?;
    Ok(policy)
}

fn load_with_fallback<T, F>(active_path: &str, fallback_path: &str, parser: F) -> Result<T>
where
    F: Fn(&str) -> Result<T>,
{
    if Path::new(active_path).exists() {
        return parser(active_path);
    }
    parser(fallback_path)
}

fn validate_kb_feature_count(kb: &RetrievalKB, expected_count: usize) -> Result<()> {
    if kb.feature_names.len() != expected_count {
        return Err(anyhow!(
            "Retrieval KB feature count mismatch: kb has {}, model expects {}",
            kb.feature_names.len(),
            expected_count
        ));
    }
    Ok(())
}

fn maybe_reload(
    kb: &mut RetrievalKB,
    policy: &mut PolicyConfig,
    sync_dir: &str,
    expected_count: usize,
) -> Result<()> {
    let trigger = format!("{sync_dir}/.reload_trigger");
    println!("Checking for reload trigger at: {}", trigger);
    if !Path::new(&trigger).exists() {
        return Ok(());
    }

    let kb_path = format!("{sync_dir}/active/kb.json");
    let policy_path = format!("{sync_dir}/active/policy.json");

    let new_kb = load_kb(&kb_path)?;
    validate_kb_feature_count(&new_kb, expected_count)?;
    let new_policy = load_policy(&policy_path)?;

    *kb = new_kb;
    *policy = new_policy;

    fs::remove_file(&trigger)?;
    println!("Reloaded KB and policy from sync state");

    Ok(())
}

fn fuse_trust(
    model_trust: f64,
    retrieval_trust: f64,
    history_score: f64,
    policy: &PolicyConfig,
) -> (f64, f64, f64, f64) {
    let gates = &policy.confidence_gates;
    let band = if model_trust >= gates.high || model_trust <= (1.0 - gates.high) {
        &policy.fusion.high_confidence
    } else if model_trust >= gates.medium
        || model_trust <= gates.low_center_min
        || model_trust <= (1.0 - gates.high_center_max)
    {
        &policy.fusion.medium_confidence
    } else {
        &policy.fusion.low_confidence
    };

    let final_trust =
        band.w_m * model_trust + band.w_r * retrieval_trust + band.w_h * history_score;
    (final_trust, band.w_m, band.w_r, band.w_h)
}

fn policy_action(trust: f64, t: &Thresholds) -> i32 {
    if trust >= t.allow {
        0
    } else if trust >= t.limit {
        1
    } else if trust >= t.quarantine {
        2
    } else {
        3
    }
}

fn write_sample_to_wasm_memory(
    memory: &wasmtime::Memory,
    store: &mut Store<()>,
    ptr: usize,
    values: &[f64],
) -> Result<()> {
    for (i, value) in values.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let offset = ptr + i * 8;
        memory.write(&mut *store, offset, &bytes)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let wasm_path = "../wasm_module/target/wasm32-unknown-unknown/release/wasm_module.wasm";
    let sequence_path = "../results/models/sequence_input.json";
    let default_kb_path = "../results/models/retrieval_kb.json";
    let default_policy_path = "../results/models/policy.json";
    let sync_dir = "../sync_state";
    let sync_active_kb_path = "../sync_state/active/kb.json";
    let sync_active_policy_path = "../sync_state/active/policy.json";
    let demo_step_sleep_ms = env::var("DEMO_STEP_SLEEP_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| anyhow!("Wasm memory export not found"))?;

    let get_ptr = instance.get_typed_func::<(), i32>(&mut store, "get_input_buffer_ptr")?;
    let get_count =
        instance.get_typed_func::<(), i32>(&mut store, "get_expected_feature_count")?;
    let attack_fn =
        instance.get_typed_func::<(), i32>(&mut store, "predict_attack_probability_x1000000")?;
    let trust_fn =
        instance.get_typed_func::<(), i32>(&mut store, "predict_trust_score_x1000000")?;

    let sequence_str = fs::read_to_string(sequence_path)?;
    let sequence: Vec<SequenceSample> = serde_json::from_str(&sequence_str)?;

    let expected_count = get_count.call(&mut store, ())? as usize;

    let mut kb = load_with_fallback(sync_active_kb_path, default_kb_path, load_kb)?;
    let mut policy =
        load_with_fallback(sync_active_policy_path, default_policy_path, load_policy)?;

    if sequence.is_empty() {
        return Err(anyhow!("Sequence is empty"));
    }

    validate_kb_feature_count(&kb, expected_count)?;

    for sample in &sequence {
        if sample.values.len() != expected_count {
            return Err(anyhow!(
                "Feature count mismatch: sample has {}, model expects {}",
                sample.values.len(),
                expected_count
            ));
        }
    }

    let ptr = get_ptr.call(&mut store, ())? as usize;

    println!("=== Retrieval-Augmented Trust Simulation (Calibrated) ===");
    println!("Feature count: {}", expected_count);

    let mut trust_history: Vec<f64> = vec![];
    let max_history = 5;

    for (step, sample) in sequence.iter().enumerate() {
        maybe_reload(&mut kb, &mut policy, sync_dir, expected_count)?;

        write_sample_to_wasm_memory(&memory, &mut store, ptr, &sample.values)?;

        let attack_prob_raw = attack_fn.call(&mut store, ())?;
        let trust_score_raw = trust_fn.call(&mut store, ())?;

        let attack_probability = attack_prob_raw as f64 / 1_000_000.0;
        let model_trust = trust_score_raw as f64 / 1_000_000.0;

        let model_action = if model_trust >= 0.80 {
            "ALLOW"
        } else if model_trust >= 0.50 {
            "LIMIT"
        } else if model_trust >= 0.20 {
            "QUARANTINE"
        } else {
            "BLOCK"
        };

        let history_score = compute_history_score(&trust_history, model_trust);
        let (retrieval_trust, profile_name, similarity, profile_trust) =
            compute_retrieval_score(&sample.values, &kb);

        let (final_trust, w_m, w_r, w_h) =
            fuse_trust(model_trust, retrieval_trust, history_score, &policy);

        trust_history.push(model_trust);
        if trust_history.len() > max_history {
            trust_history.remove(0);
        }

        let final_action_code = policy_action(final_trust, &policy.thresholds);

        println!("\n--- Time Step t{} ---", step + 1);
        println!("Attack probability : {:.6}", attack_probability);

        println!("Model trust M(x)   : {:.6}", model_trust);
        println!("Model action       : {}", model_action);

        println!("History score H(x) : {:.6}", history_score);

        println!("Retrieval profile  : {}", profile_name);
        println!("Profile trust      : {:.6}", profile_trust);
        println!("Similarity         : {:.6}", similarity);
        println!("Retrieval trust R(x): {:.6}", retrieval_trust);

        println!(
            "Fusion weights      : M={:.2}, R={:.2}, H={:.2}",
            w_m, w_r, w_h
        );

        println!("Final trust T(x)   : {:.6}", final_trust);
        println!("Final action       : {}", action_name(final_action_code));

        println!("History window     : {:?}", trust_history);
        println!("(Reference trust   : {:.6})", sample.trust);
        println!(
            "(Attack prob delta : {:.6})",
            (attack_probability - sample.attack_probability).abs()
        );

        if demo_step_sleep_ms > 0 {
            thread::sleep(Duration::from_millis(demo_step_sleep_ms));
        }
    }

    Ok(())
}