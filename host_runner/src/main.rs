use anyhow::{anyhow, Result};
use std::fs;
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
struct SampleInput {
    feature_names: Vec<String>,
    values: Vec<f64>,
    expected_attack_probability: f64,
    expected_trust_score: f64,
}

fn compute_history_score(history: &[f64], current: f64) -> f64 {
    if history.is_empty() {
        return current;
    }

    let avg = history.iter().sum::<f64>() / history.len() as f64;
    let deviation = (current - avg).abs();

    let mut score = 1.0 - deviation;

    if score < 0.0 {
        score = 0.0;
    }
    if score > 1.0 {
        score = 1.0;
    }

    score
}

fn fuse_trust(model_trust: f64, history_score: f64) -> f64 {
    0.7 * model_trust + 0.3 * history_score
}

fn main() -> Result<()> {
    let wasm_path = "../wasm_module/target/wasm32-unknown-unknown/release/wasm_module.wasm";
    let sample_path = "../results/models/numeric_dt_sample_input.json";

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
    let _action_fn = instance.get_typed_func::<(), i32>(&mut store, "predict_action_code")?;

    let sample_str = fs::read_to_string(sample_path)?;
    let sample: SampleInput = serde_json::from_str(&sample_str)?;

    let expected_count = get_count.call(&mut store, ())? as usize;
    if sample.values.len() != expected_count {
        return Err(anyhow!(
            "Feature count mismatch: sample has {}, model expects {}",
            sample.values.len(),
            expected_count
        ));
    }

    let ptr = get_ptr.call(&mut store, ())? as usize;

    // Write f64 values into Wasm memory once
    for (i, value) in sample.values.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let offset = ptr + i * 8;
        memory.write(&mut store, offset, &bytes)?;
    }

    // Simulated recent trust history
    let mut trust_history: Vec<f64> = vec![0.91, 0.88, 0.86, 0.83];
    let max_history = 5;

    // Warm-up
    let _ = attack_fn.call(&mut store, ())?;
    let _ = trust_fn.call(&mut store, ())?;

    // Current model output
    let attack_prob_raw = attack_fn.call(&mut store, ())?;
    let trust_score_raw = trust_fn.call(&mut store, ())?;

    let attack_probability = attack_prob_raw as f64 / 1_000_000.0;
    let model_trust = trust_score_raw as f64 / 1_000_000.0;

    // Behavior-memory layer
    let history_score = compute_history_score(&trust_history, model_trust);
    let final_trust = fuse_trust(model_trust, history_score);

    // Update history after computing current decision
    trust_history.push(model_trust);
    if trust_history.len() > max_history {
        trust_history.remove(0);
    }

    let final_action_code = if final_trust >= 0.80 {
        0
    } else if final_trust >= 0.50 {
        1
    } else if final_trust >= 0.20 {
        2
    } else {
        3
    };

    println!("=== Behavior-Aware Trust Inference ===");
    println!("Feature count            : {}", sample.values.len());

    println!("\n--- Model Output ---");
    println!("Attack probability       : {:.6}", attack_probability);
    println!("Model trust (M(x))       : {:.6}", model_trust);

    println!("\n--- Behavior Layer ---");
    println!("History (last N)         : {:?}", trust_history);
    println!("History score (H(x))     : {:.6}", history_score);

    println!("\n--- Final Decision ---");
    println!("Final trust (T(x))       : {:.6}", final_trust);
    println!("Policy action            : {}", action_name(final_action_code));

    println!("\n--- Reference ---");
    println!("Expected attack prob     : {:.6}", sample.expected_attack_probability);
    println!("Expected trust score     : {:.6}", sample.expected_trust_score);
    println!(
        "Attack probability delta : {:.6}",
        (attack_probability - sample.expected_attack_probability).abs()
    );

    Ok(())
}