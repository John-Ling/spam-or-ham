use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::sync::LazyLock;

const EMAIL_REGEX: &str = r"[a-zA-Z.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*";
const EMOTICON_REGEX: &str = r"[:;=8]-?[)(DPOpo|\\/*]|\([a-z]{2,10}\)";

static TOKEN_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"{EMAIL_REGEX}|{EMOTICON_REGEX}|\w+|[[:punct:]]+")).unwrap()
});

#[derive(Serialize, Deserialize, Debug)]
pub struct Classifier {
    alpha: f64,
    content_column: String,
    label_column: String,
    negative_label: String,
    positive_label: String,
    positive_probability: f64,
    positive_likelihoods: HashMap<String, f64>,
    negative_likelihoods: HashMap<String, f64>,
    positive_unseen_likelihood: f64,
    negative_unseen_likelihood: f64,
}
impl Classifier {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            content_column: String::from(""),
            label_column: String::from(""),
            negative_label: String::from(""),
            positive_label: String::from(""),
            positive_probability: 0.0,
            positive_likelihoods: HashMap::new(),
            negative_likelihoods: HashMap::new(),
            positive_unseen_likelihood: 0.0,
            negative_unseen_likelihood: 0.0,
        }
    }

    pub fn load_pretrained(&mut self, model_string: &str) -> Result<(), Box<dyn Error>> {
        let model = serde_json::from_str::<Classifier>(model_string)?;
        self.alpha = model.alpha;
        self.content_column = model.content_column;
        self.label_column = model.label_column;
        self.negative_label = model.negative_label;
        self.positive_label = model.positive_label;
        self.positive_probability = model.positive_probability;
        self.positive_likelihoods = model.positive_likelihoods;
        self.positive_unseen_likelihood = model.positive_unseen_likelihood;
        self.negative_likelihoods = model.negative_likelihoods;
        self.negative_unseen_likelihood = model.negative_unseen_likelihood;
        Ok(())
    }

    fn preprocess_text(text: &str) -> String {
        TOKEN_REGEX
            .find_iter(text)
            .map(|m| m.as_str().to_lowercase())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn predict(&self, text: &str) -> Result<(bool, Vec<(f64, f64)>), Box<dyn Error>> {
        let preprocessed = Self::preprocess_text(text);
        let tokens: Vec<&str> = preprocessed.split_whitespace().collect();
        let mut positive_probability = self.positive_probability.ln();
        let mut negative_probability = (1.0 - self.positive_probability).ln();

        let mut probabilities = vec![(0.0, 0.0); tokens.len()];
        for (i, t) in tokens.iter().enumerate() {
            let token = *t;
            let lp = self.positive_likelihoods.get(token);
            let ln = self.negative_likelihoods.get(token);

            // Skip words that appear in neither class
            if ln.is_none() && lp.is_none() {
                continue;
            }

            let likelihood_positive = lp.unwrap_or(&self.positive_unseen_likelihood);
            let likelihood_negative = ln.unwrap_or(&self.negative_unseen_likelihood);

            probabilities[i] = (likelihood_positive.clone(), likelihood_negative.clone());

            // Multiplying small probabilities can lead to underflow hence convert to log scale to and sum instead
            positive_probability += likelihood_positive.ln();
            negative_probability += likelihood_negative.ln();
        }

        // Normalisation via logistic sigmoid
        let p = 1.0 / (1.0 + (negative_probability - positive_probability).exp());
        Ok((p > 0.995, probabilities))
    }
}
