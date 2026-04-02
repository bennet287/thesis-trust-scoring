use serde::Deserialize;

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

fn trust_to_action_code(trust_score: f64) -> i32 {
    if trust_score >= 0.80 {
        0 // ALLOW
    } else if trust_score >= 0.50 {
        1 // LIMIT
    } else if trust_score >= 0.20 {
        2 // QUARANTINE
    } else {
        3 // BLOCK
    }
}

fn load_model() -> TreeExport {
    let model_str = include_str!("../../results/models/numeric_dt_export.json");
    serde_json::from_str(model_str).expect("Could not parse embedded model JSON")
}

fn load_sample() -> SampleInput {
    let sample_str = include_str!("../../results/models/numeric_dt_sample_input.json");
    serde_json::from_str(sample_str).expect("Could not parse embedded sample JSON")
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

        if left == -1 && right == -1 {
            let counts = &tree.value[node][0];
            let total: f64 = counts.iter().sum();
            if total <= 0.0 {
                return 0.0;
            }
            return counts[1] / total; // class 1 = attack
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

#[unsafe(no_mangle)]
pub extern "C" fn predict_demo_attack_probability_x1000000() -> i32 {
    let model = load_model();
    let sample = load_sample();
    let attack_probability = predict_proba(&sample.values, &model);
    (attack_probability * 1_000_000.0).round() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn predict_demo_trust_score_x1000000() -> i32 {
    let model = load_model();
    let sample = load_sample();
    let attack_probability = predict_proba(&sample.values, &model);
    let trust_score = 1.0 - attack_probability;
    (trust_score * 1_000_000.0).round() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn predict_demo_action_code() -> i32 {
    let model = load_model();
    let sample = load_sample();
    let attack_probability = predict_proba(&sample.values, &model);
    let trust_score = 1.0 - attack_probability;
    trust_to_action_code(trust_score)
}

#[unsafe(no_mangle)]
pub extern "C" fn expected_attack_probability_x1000000() -> i32 {
    let sample = load_sample();
    (sample.expected_attack_probability * 1_000_000.0).round() as i32
}