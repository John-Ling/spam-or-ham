use wasm_bindgen::prelude::*;
mod classifier;
use classifier::Classifier;
use std::sync::OnceLock;

static CLASSIFIER_INSTANCE: OnceLock<Classifier> = OnceLock::new();

#[wasm_bindgen]
pub fn load_classifier(model_json_string: &str) -> bool {
    let mut classifier = Classifier::new();

    let successful = classifier.load_pretrained(model_json_string).is_ok();
    CLASSIFIER_INSTANCE.set(classifier).ok();
    return successful;
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct PredictionResult {
    pub outcome: bool,
    positive_probabilities: Vec<f64>,
    negative_probabilities: Vec<f64>,
}

#[wasm_bindgen]
impl PredictionResult {
    #[wasm_bindgen(getter)]
    pub fn positive_probabilities(&self) -> Vec<f64> {
        self.positive_probabilities.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn negative_probabilities(&self) -> Vec<f64> {
        self.negative_probabilities.clone()
    }
}

#[wasm_bindgen]
pub fn predict(string: &str) -> PredictionResult {
    let classifier = CLASSIFIER_INSTANCE.get().unwrap();
    let (outcome, probabilities) = classifier.predict(string).unwrap();

    // Split tuples into two vectors because for some fucking reason Typescript can't read tuples from Rust
    let positive_probabilities: Vec<f64> = (0..probabilities.len())
        .map(|i| probabilities[i].0)
        .collect();
    let negative_probabilities: Vec<f64> = (0..probabilities.len())
        .map(|i| probabilities[i].1)
        .collect();

    return PredictionResult {
        outcome: outcome,
        positive_probabilities: positive_probabilities,
        negative_probabilities: negative_probabilities,
    };
}
