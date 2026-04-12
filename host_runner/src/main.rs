use anyhow::{anyhow, Result};
use std::env;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;
use wasmtime::{Engine, Instance, Module, Store};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeploymentProfile {
    Full,
    Compact,
}

impl DeploymentProfile {
    fn from_env() -> Self {
        match env::var("LTCA_PROFILE")
            .unwrap_or_else(|_| "full".to_string())
            .to_lowercase()
            .as_str()
        {
            "compact" => DeploymentProfile::Compact,
            _ => DeploymentProfile::Full,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            DeploymentProfile::Full => "full",
            DeploymentProfile::Compact => "compact",
        }
    }
}

fn action_name(code: i32) -> &'static str {
    match code {
        0 => "ALLOW",
        1 => "LIMIT",
        2 => "QUARANTINE",
        3 => "BLOCK",
        _ => "UNKNOWN",
    }
}

fn action_to_code(action_code: i32) -> u8 {
    match action_code {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        _ => 0,
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

#[derive(Debug, Clone)]
struct FullHistory {
    window: Vec<f64>,
    max_len: usize,
}

impl FullHistory {
    fn new(max_len: usize) -> Self {
        Self {
            window: Vec::with_capacity(max_len),
            max_len,
        }
    }

    fn update(&mut self, current: f64) -> f64 {
        let score = compute_history_score(&self.window, current);
        self.window.push(current);
        if self.window.len() > self.max_len {
            self.window.remove(0);
        }
        score
    }
}

#[derive(Debug, Clone)]
struct CompactHistory {
    ema_trust: f32,
    variance: f32,
    anomaly_count: u8,
    last_action: u8,
    seconds_since_safe: u16,
    initialized: bool,
}

impl CompactHistory {
    fn new() -> Self {
        Self {
            ema_trust: 1.0,
            variance: 0.0,
            anomaly_count: 0,
            last_action: 0,
            seconds_since_safe: 0,
            initialized: false,
        }
    }

    fn update(&mut self, trust: f32, action_code: u8, delta_seconds: u16) -> f32 {
        let alpha = 0.3_f32;

        if !self.initialized {
            self.ema_trust = trust;
            self.variance = 0.0;
            self.initialized = true;
        } else {
            let diff = trust - self.ema_trust;
            self.ema_trust = (1.0 - alpha) * self.ema_trust + alpha * trust;
            self.variance = (1.0 - alpha) * self.variance + alpha * diff * diff;
        }

        if trust < 0.5 {
            self.anomaly_count = self.anomaly_count.saturating_add(1);
        } else {
            // Recovery is gradual to avoid instantly forgetting unstable periods.
            self.anomaly_count = self.anomaly_count.saturating_sub(1);
        }

        self.last_action = action_code;

        if trust >= 0.8 {
            self.seconds_since_safe = 0;
        } else {
            self.seconds_since_safe = self.seconds_since_safe.saturating_add(delta_seconds);
        }

        self.compute_h_score()
    }

    fn compute_h_score(&self) -> f32 {
        let anomaly_penalty_base = (self.anomaly_count as f32 * 0.05).min(0.25);
        let anomaly_penalty_extra = if self.anomaly_count >= 3 { 0.08 } else { 0.0 };
        let anomaly_penalty = (anomaly_penalty_base + anomaly_penalty_extra).min(0.35);

        // Ignore tiny variance so stable operation is not over-penalized.
        let variance_penalty = if self.variance > 0.02 {
            (self.variance - 0.02).min(0.20)
        } else {
            0.0
        };

        let action_penalty = if self.last_action >= 2 {
            0.06
        } else if self.last_action == 1 {
            0.02
        } else {
            0.0
        };

        let staleness_penalty = ((self.seconds_since_safe as f32) / 3600.0 * 0.05).min(0.15);

        let recovery_boost = if self.anomaly_count == 0
            && self.variance < 0.02
            && self.seconds_since_safe <= 10
        {
            0.04
        } else {
            0.0
        };

        let score = self.ema_trust
            - anomaly_penalty
            - variance_penalty
            - action_penalty
            - staleness_penalty
            + recovery_boost;
        score.clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone)]
enum HistoryState {
    Full(FullHistory),
    Compact(CompactHistory),
}

impl HistoryState {
    fn new(profile: DeploymentProfile, max_history: usize) -> Self {
        match profile {
            DeploymentProfile::Full => HistoryState::Full(FullHistory::new(max_history)),
            DeploymentProfile::Compact => HistoryState::Compact(CompactHistory::new()),
        }
    }

    fn update(&mut self, model_trust: f64, action_code: u8, delta_seconds: u16) -> f64 {
        match self {
            HistoryState::Full(h) => h.update(model_trust),
            HistoryState::Compact(h) => h.update(model_trust as f32, action_code, delta_seconds) as f64,
        }
    }

    fn debug_repr(&self) -> String {
        match self {
            HistoryState::Full(h) => format!("{:?}", h.window),
            HistoryState::Compact(h) => format!(
                "compact{{ema={:.4}, var={:.4}, anomalies={}, last_action={}, safe_age_s={}}}",
                h.ema_trust, h.variance, h.anomaly_count, h.last_action, h.seconds_since_safe
            ),
        }
    }
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

fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn compute_retrieval_score(values: &[f64], kb: &RetrievalKB) -> (f64, String, f64, f64, f64) {
    let x_norm = normalize_with_minmax(values, &kb.scaler_min, &kb.scaler_max);

    let mut best_profile: Option<&RetrievalProfile> = None;
    let mut best_distance = f64::INFINITY;
    let mut second_best_distance = f64::INFINITY;

    for profile in &kb.profiles {
        let dist = euclidean_distance(&x_norm, &profile.centroid);
        if dist < best_distance {
            second_best_distance = best_distance;
            best_distance = dist;
            best_profile = Some(profile);
        } else if dist < second_best_distance {
            second_best_distance = dist;
        }
    }

    let profile = best_profile.expect("Retrieval KB has no profiles");

    let max_dist = (x_norm.len() as f64).sqrt();
    let similarity = (1.0 - best_distance / max_dist).clamp(0.0, 1.0);
    let margin = if second_best_distance.is_finite() {
        ((second_best_distance - best_distance) / max_dist).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let retrieval_confidence = (similarity * (0.5 + 0.5 * margin)).clamp(0.0, 1.0);
    let retrieval_trust = retrieval_confidence * profile.profile_trust + (1.0 - retrieval_confidence) * 0.5;

    (
        retrieval_trust,
        profile.name.clone(),
        similarity,
        profile.profile_trust,
        retrieval_confidence,
    )
}

fn compute_retrieval_score_compact(values: &[f64], kb: &RetrievalKB) -> (f64, String, f64, f64, f64) {
    let x_norm64 = normalize_with_minmax(values, &kb.scaler_min, &kb.scaler_max);
    let x_norm: Vec<f32> = x_norm64.iter().map(|v| *v as f32).collect();

    let mut best_profile: Option<&RetrievalProfile> = None;
    let mut best_distance = f32::INFINITY;
    let mut second_best_distance = f32::INFINITY;

    for profile in &kb.profiles {
        let centroid: Vec<f32> = profile.centroid.iter().map(|v| *v as f32).collect();
        let dist = euclidean_distance_f32(&x_norm, &centroid);
        if dist < best_distance {
            second_best_distance = best_distance;
            best_distance = dist;
            best_profile = Some(profile);
        } else if dist < second_best_distance {
            second_best_distance = dist;
        }
    }

    let profile = best_profile.expect("Retrieval KB has no profiles");

    let max_dist = (x_norm.len() as f32).sqrt();
    let similarity = (1.0_f32 - best_distance / max_dist).clamp(0.0, 1.0);
    let margin = if second_best_distance.is_finite() {
        ((second_best_distance - best_distance) / max_dist).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let retrieval_confidence = (similarity * (0.5 + 0.5 * margin)).clamp(0.0, 1.0);
    let profile_trust = profile.profile_trust as f32;
    let retrieval_trust = retrieval_confidence * profile_trust + (1.0 - retrieval_confidence) * 0.5;

    (
        retrieval_trust as f64,
        profile.name.clone(),
        similarity as f64,
        profile.profile_trust,
        retrieval_confidence as f64,
    )
}

fn load_kb(path: &str) -> Result<RetrievalKB> {
    let kb_str = fs::read_to_string(path)?;
    let kb: RetrievalKB = serde_json::from_str(&kb_str)?;
    Ok(kb)
}

fn load_kb_cbor(path: &str) -> Result<RetrievalKB> {
    let raw = fs::read(path)?;
    let kb: RetrievalKB = ciborium::from_reader(raw.as_slice())?;
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
    profile: DeploymentProfile,
) -> Result<()> {
    let trigger = format!("{sync_dir}/.reload_trigger");
    println!("Checking for reload trigger at: {}", trigger);
    if !Path::new(&trigger).exists() {
        return Ok(());
    }

    let kb_json_path = format!("{sync_dir}/active/kb.json");
    let kb_cbor_path = format!("{sync_dir}/active/kb.cbor");
    let policy_path = format!("{sync_dir}/active/policy.json");

    let new_kb = match profile {
        DeploymentProfile::Full => load_kb(&kb_json_path)?,
        DeploymentProfile::Compact => {
            if Path::new(&kb_cbor_path).exists() {
                load_kb_cbor(&kb_cbor_path)?
            } else {
                load_kb(&kb_json_path)?
            }
        }
    };

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
    retrieval_confidence: f64,
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

    let mut w_m = band.w_m;
    let mut w_r = band.w_r;
    let mut w_h = band.w_h;

    // When retrieval is ambiguous, shift part of its influence to model/history.
    if retrieval_confidence < 0.55 {
        let shift_ratio = ((0.55 - retrieval_confidence) / 0.55).clamp(0.0, 1.0);
        let delta = w_r * 0.5 * shift_ratio;
        w_r -= delta;
        w_m += delta * 0.7;
        w_h += delta * 0.3;
    }

    // If history is unstable/low, cap its influence and bias back to model trust.
    if history_score < 0.35 {
        let delta = w_h * 0.2;
        w_h -= delta;
        w_m += delta;
    }

    let weight_sum = w_m + w_r + w_h;
    if weight_sum > 0.0 {
        w_m /= weight_sum;
        w_r /= weight_sum;
        w_h /= weight_sum;
    }

    let final_trust = w_m * model_trust + w_r * retrieval_trust + w_h * history_score;
    (final_trust, w_m, w_r, w_h)
}

fn policy_action_raw(trust: f64, t: &Thresholds) -> i32 {
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

fn policy_action_with_hysteresis(trust: f64, t: &Thresholds, previous_action: u8) -> i32 {
    let desired = policy_action_raw(trust, t);

    const ALLOW_EXIT_MARGIN: f64 = 0.03;
    const LIMIT_EXIT_MARGIN: f64 = 0.04;
    const QUARANTINE_EXIT_MARGIN: f64 = 0.05;

    match previous_action {
        // Keep LIMIT until trust is safely above ALLOW threshold.
        1 if desired == 0 && trust < (t.allow + ALLOW_EXIT_MARGIN) => 1,
        // Keep QUARANTINE until trust is safely above LIMIT threshold.
        2 if desired <= 1 && trust < (t.limit + LIMIT_EXIT_MARGIN) => 2,
        // Keep BLOCK until trust is safely above QUARANTINE threshold.
        3 if desired <= 2 && trust < (t.quarantine + QUARANTINE_EXIT_MARGIN) => 3,
        _ => desired,
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
    let default_kb_cbor_path = "../results/models/retrieval_kb.cbor";
    let default_policy_path = "../results/models/policy.json";
    let sync_dir = "../sync_state";
    let sync_active_kb_path = "../sync_state/active/kb.json";
    let sync_active_kb_cbor_path = "../sync_state/active/kb.cbor";
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

    let profile = DeploymentProfile::from_env();
    println!("LTCA deployment profile: {}", profile.as_str());

    let mut kb = match profile {
        DeploymentProfile::Full => load_with_fallback(sync_active_kb_path, default_kb_path, load_kb)?,
        DeploymentProfile::Compact => {
            if Path::new(sync_active_kb_cbor_path).exists() {
                load_kb_cbor(sync_active_kb_cbor_path)?
            } else if Path::new(sync_active_kb_path).exists() {
                load_kb(sync_active_kb_path)?
            } else if Path::new(default_kb_cbor_path).exists() {
                load_kb_cbor(default_kb_cbor_path)?
            } else {
                load_kb(default_kb_path)?
            }
        }
    };

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

    let mut history = HistoryState::new(profile, 5);
    let mut previous_action_code: u8 = 0;

    for (step, sample) in sequence.iter().enumerate() {
        maybe_reload(&mut kb, &mut policy, sync_dir, expected_count, profile)?;

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

        let history_score = history.update(model_trust, previous_action_code, 1);
        let (retrieval_trust, profile_name, similarity, profile_trust, retrieval_confidence) = match profile {
            DeploymentProfile::Full => compute_retrieval_score(&sample.values, &kb),
            DeploymentProfile::Compact => compute_retrieval_score_compact(&sample.values, &kb),
        };

        let (final_trust, w_m, w_r, w_h) =
            fuse_trust(model_trust, retrieval_trust, retrieval_confidence, history_score, &policy);

        let final_action_code =
            policy_action_with_hysteresis(final_trust, &policy.thresholds, previous_action_code);
        previous_action_code = action_to_code(final_action_code);

        println!("\n--- Time Step t{} ---", step + 1);
        println!("Attack probability : {:.6}", attack_probability);

        println!("Model trust M(x)   : {:.6}", model_trust);
        println!("Model action       : {}", model_action);

        println!("History score H(x) : {:.6}", history_score);

        println!("Retrieval profile  : {}", profile_name);
        println!("Profile trust      : {:.6}", profile_trust);
        println!("Similarity         : {:.6}", similarity);
        println!("Retrieval confidence: {:.6}", retrieval_confidence);
        println!("Retrieval trust R(x): {:.6}", retrieval_trust);

        println!(
            "Fusion weights      : M={:.2}, R={:.2}, H={:.2}",
            w_m, w_r, w_h
        );

        println!("Final trust T(x)   : {:.6}", final_trust);
        println!("Final action       : {}", action_name(final_action_code));

        println!("History state      : {}", history.debug_repr());
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