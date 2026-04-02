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
    let action_fn = instance.get_typed_func::<(), i32>(&mut store, "predict_action_code")?;

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

    // write f64 values into Wasm memory
    for (i, value) in sample.values.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let offset = ptr + i * 8;
        memory.write(&mut store, offset, &bytes)?;
    }

    let attack_prob_raw = attack_fn.call(&mut store, ())?;
    let trust_score_raw = trust_fn.call(&mut store, ())?;
    let action_code = action_fn.call(&mut store, ())?;

    let attack_probability = attack_prob_raw as f64 / 1_000_000.0;
    let trust_score = trust_score_raw as f64 / 1_000_000.0;

    println!("=== Generalized Wasm Inference ===");
    println!("Feature count            : {}", sample.values.len());
    println!("Attack probability       : {:.6}", attack_probability);
    println!("Trust score              : {:.6}", trust_score);
    println!("Policy action            : {}", action_name(action_code));
    println!("Expected attack prob     : {:.6}", sample.expected_attack_probability);
    println!("Expected trust score     : {:.6}", sample.expected_trust_score);
    println!(
        "Attack probability delta : {:.6}",
        (attack_probability - sample.expected_attack_probability).abs()
    );
    println!(
        "Trust score delta        : {:.6}",
        (trust_score - sample.expected_trust_score).abs()
    );

    Ok(())
}