use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TreeExport {
    feature_names: Vec<String>,
    imputer_statistics: Vec<f64>,
    scaler_min: Vec<f64>,
    scaler_max: Vec<f64>,
    tree: TreeData,
}

#[derive(Debug, Deserialize)]
struct TreeData {
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    value: Vec<Vec<Vec<f64>>>,
    node_count: usize,
    max_depth: usize,
    n_classes: usize,
    classes: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct SampleInput {
    feature_names: Vec<String>,
    values: Vec<f64>,
    expected_attack_probability: f64,
    expected_trust_score: f64,
}

fn trust_to_action(trust_score: f64) -> &'static str {
    if trust_score >= 0.80 {
        "ALLOW"
    } else if trust_score >= 0.50 {
        "LIMIT"
    } else if trust_score >= 0.20 {
        "QUARANTINE"
    } else {
        "BLOCK"
    }
}

fn impute_and_scale(values: &[f64], model: &TreeExport) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let mut v = x;
            if v.is_nan() {
                v = model.imputer_statistics[i];
            }

            let min = model.scaler_min[i];
            let max = model.scaler_max[i];

            if (max - min).abs() < 1e-12 {
                0.0
            } else {
                (v - min) / (max - min)
            }
        })
        .collect()
}

fn predict_proba(values: &[f64], model: &TreeExport) -> f64 {
    let x = impute_and_scale(values, model);
    let tree = &model.tree;
    let mut node: usize = 0;

    loop {
        let left = tree.children_left[node];
        let right = tree.children_right[node];

        // Leaf node
        if left == -1 && right == -1 {
            let counts = &tree.value[node][0];
            let total: f64 = counts.iter().sum();
            if total <= 0.0 {
                return 0.0;
            }
            // Probability of class 1 = attack
            return counts[1] / total;
        }

        let feature_idx = tree.feature[node] as usize;
        let threshold = tree.threshold[node];

        if x[feature_idx] <= threshold {
            node = left as usize;
        } else {
            node = right as usize;
        }
    }
}

fn main() {
    let model_path = Path::new("../results/models/numeric_dt_export.json");
    let sample_path = Path::new("../results/models/numeric_dt_sample_input.json");

    let model_str =
        fs::read_to_string(model_path).expect("Could not read numeric_dt_export.json");
    let model: TreeExport =
        serde_json::from_str(&model_str).expect("Could not parse model JSON");

    let sample_str =
        fs::read_to_string(sample_path).expect("Could not read numeric_dt_sample_input.json");
    let sample: SampleInput =
        serde_json::from_str(&sample_str).expect("Could not parse sample JSON");

    let attack_probability = predict_proba(&sample.values, &model);
    let trust_score = 1.0 - attack_probability;
    let action = trust_to_action(trust_score);

    println!("=== Rust Numeric DT Inference ===");
    println!("Feature count            : {}", sample.values.len());
    println!("Attack probability       : {:.6}", attack_probability);
    println!("Trust score              : {:.6}", trust_score);
    println!("Policy action            : {}", action);
    println!("Expected attack prob     : {:.6}", sample.expected_attack_probability);
    println!("Expected trust score     : {:.6}", sample.expected_trust_score);
    println!(
        "Attack probability delta : {:.6}",
        (attack_probability - sample.expected_attack_probability).abs()
    );
}