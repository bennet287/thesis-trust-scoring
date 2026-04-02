use anyhow::Result;
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

fn main() -> Result<()> {
    let wasm_path = "../wasm_module/target/wasm32-unknown-unknown/release/wasm_module.wasm";

    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let attack_fn = instance.get_typed_func::<(), i32>(
        &mut store,
        "predict_demo_attack_probability_x1000000",
    )?;
    let trust_fn = instance.get_typed_func::<(), i32>(
        &mut store,
        "predict_demo_trust_score_x1000000",
    )?;
    let action_fn =
        instance.get_typed_func::<(), i32>(&mut store, "predict_demo_action_code")?;
    let expected_fn = instance.get_typed_func::<(), i32>(
        &mut store,
        "expected_attack_probability_x1000000",
    )?;

    let attack_prob_raw = attack_fn.call(&mut store, ())?;
    let trust_score_raw = trust_fn.call(&mut store, ())?;
    let action_code = action_fn.call(&mut store, ())?;
    let expected_raw = expected_fn.call(&mut store, ())?;

    let attack_probability = attack_prob_raw as f64 / 1_000_000.0;
    let trust_score = trust_score_raw as f64 / 1_000_000.0;
    let expected_attack = expected_raw as f64 / 1_000_000.0;

    println!("=== Wasm Demo Inference ===");
    println!("Attack probability       : {:.6}", attack_probability);
    println!("Trust score              : {:.6}", trust_score);
    println!("Policy action            : {}", action_name(action_code));
    println!("Expected attack prob     : {:.6}", expected_attack);
    println!(
        "Attack probability delta : {:.6}",
        (attack_probability - expected_attack).abs()
    );

    Ok(())
}