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
    positive_likelihoods: Vec<f64>,
    negative_likelihoods: Vec<f64>,
    pub positive_probability: f64,
    pub negative_probability: f64,
}

#[wasm_bindgen]
impl PredictionResult {
    #[wasm_bindgen(getter)]
    pub fn positive_likelihoods(&self) -> Vec<f64> {
        self.positive_likelihoods.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn negative_likelihoods(&self) -> Vec<f64> {
        self.negative_likelihoods.clone()
    }
}

#[wasm_bindgen]
pub fn predict(string: &str) -> PredictionResult {
    let classifier = CLASSIFIER_INSTANCE.get().unwrap();
    let (outcome, positive_probability, negative_probability, probabilities) =
        classifier.predict(string).unwrap();

    // Split tuples into two vectors because for some fucking reason Typescript can't read tuples from Rust
    let positive_likelihoods: Vec<f64> = (0..probabilities.len())
        .map(|i| probabilities[i].0)
        .collect();
    let negative_likelihoods: Vec<f64> = (0..probabilities.len())
        .map(|i| probabilities[i].1)
        .collect();

    return PredictionResult {
        outcome: outcome,
        positive_likelihoods: positive_likelihoods,
        negative_likelihoods: negative_likelihoods,
        positive_probability: positive_probability,
        negative_probability: negative_probability,
    };
}
