use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::LazyLock;
use stop_words::{Language, get};

const TRAILING_PUNCTUATION_REGEX_: &str = r"\w+|[[:punct:]]+";
const TRAILING_PUNCTUATION_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(TRAILING_PUNCTUATION_REGEX_).unwrap());
const EMAIL_REGEX: &str = r"[a-zA-Z.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*";
const EMOTICON_REGEX: &str = r"[:;=8]-?[)(DPOpo|\\/*]|\([a-z]{2,10}\)";

static KEEP_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    {
        Regex::new(
            format!(
                r"({})|({})|({})|([a-zA-Z0-9\s!?$%'])|(.)",
                EMAIL_REGEX, EMOTICON_REGEX, TRAILING_PUNCTUATION_REGEX_
            )
            .as_str(),
        )
    }
    .unwrap()
});

const STOP_WORDS: LazyLock<HashSet<&str>> =
    LazyLock::new(|| get(Language::English).iter().map(|s| *s).collect());

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
        // Remove stop words
        let without_stop_words = Self::tokenise(text)
            .into_iter()
            .filter(|w| !STOP_WORDS.contains(w.to_lowercase().as_str()))
            .map(|w| w.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // Remove all other characters not in the KEEP_REGEX
        // KEEP_REGEX also keeps emails and emoticons that otherwise would have been stripped
        KEEP_REGEX
            .replace_all(&without_stop_words, |caps: &regex::Captures| {
                if caps.get(4).is_none() {
                    caps[0].to_string()
                } else {
                    " ".to_string()
                }
            })
            .into_owned()
    }

    pub fn predict(&self, text: &str) -> bool {
        let preprocessed = Self::preprocess_text(text);
        let tokens: Vec<&str> = preprocessed.split_whitespace().collect();
        let mut positive_probability = self.positive_probability.ln();
        let mut negative_probability = (1.0 - self.positive_probability).ln();
        for token in tokens {
            let likelihood_positive = self.positive_likelihoods.get(token);
            let likelihood_negative = self.negative_likelihoods.get(token);

            // Skip words that appear in neither class
            if likelihood_negative.is_none() && likelihood_positive.is_none() {
                continue;
            }

            positive_probability += likelihood_positive
                .unwrap_or(&self.positive_unseen_likelihood)
                .ln();

            negative_probability += likelihood_negative
                .unwrap_or(&self.negative_unseen_likelihood)
                .ln();
        }

        // Normalisation via logistic sigmoid
        let p = 1.0 / (1.0 + (negative_probability - positive_probability).exp());
        p > 0.99
    }

    // Split text into tokens including stripping trailing punctuation as their own token
    // i.e hello! becomes "hello", "!"
    fn tokenise(text: &str) -> Vec<String> {
        let t: Vec<&str> = TRAILING_PUNCTUATION_REGEX
            .find_iter(text)
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();

        let tokens = t.iter().map(|t| t.to_string()).collect::<Vec<String>>();
        // let bigrams = t
        //     .windows(2)
        //     .map(|w| format!("{}_{}", w[0], w[1]))
        //     .collect::<Vec<String>>();
        return tokens;
        // return [tokens].concat();
    }
}
