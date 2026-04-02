use anyhow::{anyhow, Result};
use std::fs;
use wasmtime::{Engine, Instance, Module, Store};

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

fn action_name_from_trust(trust: f64) -> &'static str {
    if trust >= 0.80 {
        "ALLOW"
    } else if trust >= 0.50 {
        "LIMIT"
    } else if trust >= 0.20 {
        "QUARANTINE"
    } else {
        "BLOCK"
    }
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

    // Initial write to Wasm memory
    for (i, value) in sample.values.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let offset = ptr + i * 8;
        memory.write(&mut store, offset, &bytes)?;
    }

    // Warm-up
    let _ = attack_fn.call(&mut store, ())?;
    let _ = trust_fn.call(&mut store, ())?;

    println!("=== Temporal Trust Simulation ===");
    println!("Feature count: {}", sample.feature_names.len());

    let mut trust_history: Vec<f64> = vec![0.91, 0.88, 0.86];
    let max_history = 5;

    let mut simulated_values = sample.values.clone();

    for step in 1..=5 {
        // Simulate gradual behavior drift
        for value in &mut simulated_values {
            *value *= 1.0 + (step as f64 * 0.02);
        }

        // Write updated values into Wasm memory
        for (i, value) in simulated_values.iter().enumerate() {
            let bytes = value.to_le_bytes();
            let offset = ptr + i * 8;
            memory.write(&mut store, offset, &bytes)?;
        }

        // Run Wasm inference
        let attack_prob_raw = attack_fn.call(&mut store, ())?;
        let trust_score_raw = trust_fn.call(&mut store, ())?;

        let attack_probability = attack_prob_raw as f64 / 1_000_000.0;
        let model_trust = trust_score_raw as f64 / 1_000_000.0;

        // Behavior-aware layer
        let history_score = compute_history_score(&trust_history, model_trust);
        let final_trust = fuse_trust(model_trust, history_score);
        let action = action_name_from_trust(final_trust);

        // Update trust history
        trust_history.push(model_trust);
        if trust_history.len() > max_history {
            trust_history.remove(0);
        }

        println!("\n--- Time Step t{} ---", step);
        println!("Attack probability : {:.6}", attack_probability);
        println!("Model trust M(x)   : {:.6}", model_trust);
        println!("History score H(x) : {:.6}", history_score);
        println!("Final trust T(x)   : {:.6}", final_trust);
        println!("Action             : {}", action);
        println!("History window     : {:?}", trust_history);
    }

    println!("\n--- Reference (original sample before drift) ---");
    println!(
        "Expected attack probability: {:.6}",
        sample.expected_attack_probability
    );
    println!(
        "Expected trust score      : {:.6}",
        sample.expected_trust_score
    );

    Ok(())
}