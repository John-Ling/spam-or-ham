use csv::Reader;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::LazyLock;

pub struct Corpus {
    content: Vec<String>,
    labels: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Classifier {
    positive_label: String,
    positive_probability: f64,
    positive_likelihoods: HashMap<String, f64>,
    negative_likelihoods: HashMap<String, f64>,
    positive_unseen_likelihood: f64,
    negative_unseen_likelihood: f64,
}

static TOKEN_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+|[^\w\s]").unwrap());

impl Classifier {
    pub fn new() -> Self {
        Self {
            positive_label: String::from(""),
            positive_probability: 0.0,
            positive_likelihoods: HashMap::new(),
            negative_likelihoods: HashMap::new(),
            positive_unseen_likelihood: 0.0,
            negative_unseen_likelihood: 0.0,
        }
    }

    pub fn save_model(&mut self, model_path: &str) -> Result<(), Box<dyn Error>> {
        let serialised = Classifier {
            positive_label: self.positive_label.to_string(),
            positive_probability: self.positive_probability,
            positive_likelihoods: self.positive_likelihoods.clone(),
            negative_likelihoods: self.negative_likelihoods.clone(),
            positive_unseen_likelihood: self.positive_unseen_likelihood,
            negative_unseen_likelihood: self.negative_unseen_likelihood,
        };

        let write_file = File::create(model_path)?;
        let writer = BufWriter::new(write_file);
        match serde_json::to_writer(writer, &serialised) {
            Ok(_) => println!("Saved model"),
            Err(_) => println!("Failed to save model"),
        }
        Ok(())
    }

    pub fn load_model(&mut self, model_path: &str) -> Result<(), Box<dyn Error>> {
        let read_file = File::open(model_path)?;
        let reader = BufReader::new(read_file);
        let model: Classifier = serde_json::from_reader(reader)?;
        self.positive_label = model.positive_label;
        self.positive_probability = model.positive_probability;
        self.positive_likelihoods = model.positive_likelihoods;
        self.positive_unseen_likelihood = model.positive_unseen_likelihood;
        self.negative_likelihoods = model.negative_likelihoods;
        self.negative_unseen_likelihood = model.negative_unseen_likelihood;
        Ok(())
    }

    pub fn train(&mut self, corpus: Corpus, positive_label: &str) -> Result<(), Box<dyn Error>> {
        let mut positive_record_count = 0;
        let mut negative_record_count = 0;
        let mut record_count = 0;
        let mut unique_words: HashSet<&str> = HashSet::new();

        let mut positive_word_frequencies: HashMap<String, u32> = HashMap::new();
        let mut negative_word_frequencies: HashMap<String, u32> = HashMap::new();

        if corpus.content.len() != corpus.labels.len() {
            return Err(Box::from(
                "Number of labels in corpus does not match the amount of content",
            ));
        }

        for (_, (content, label)) in corpus.content.iter().zip(corpus.labels.iter()).enumerate() {
            let tokens = Self::tokenise(content);
            let unique_tokens: HashSet<&str> = tokens.iter().copied().collect();
            unique_words.extend(unique_tokens);
            let mut word_frequencies = &mut negative_word_frequencies;
            let mut count = &mut negative_record_count;

            if *label == positive_label {
                word_frequencies = &mut positive_word_frequencies;
                count = &mut positive_record_count;
            }

            *count += 1;
            for w in tokens {
                let _ = *word_frequencies
                    .entry(String::from(w))
                    .and_modify(|f| *f += 1)
                    .or_insert(1);
            }
            record_count += 1;
        }

        let word_count: u32 = unique_words.len() as u32;
        let positive_word_count: u32 = positive_word_frequencies.values().sum();
        let negative_word_count: u32 = negative_word_frequencies.values().sum();
        let positive_probability = positive_record_count as f64 / record_count as f64;
        self.positive_probability = positive_probability;

        // Compute conditional/likelihood probabilities
        let alpha = 1.0;
        for word in unique_words {
            // Get frequency of each word in the positive and negative samples
            // then apply laplace smoothing
            let frequency_positive = positive_word_frequencies.get(word).unwrap_or(&0);
            let frequency_negative = negative_word_frequencies.get(word).unwrap_or(&0);
            self.positive_likelihoods.insert(
                word.to_string(),
                (*frequency_positive as f64 + alpha)
                    / (positive_word_count as f64 + alpha * word_count as f64),
            );

            self.negative_likelihoods.insert(
                word.to_string(),
                (*frequency_negative as f64 + alpha)
                    / (negative_word_count as f64 + alpha * word_count as f64),
            );
        }

        self.positive_unseen_likelihood =
            alpha as f64 / (positive_word_count as f64 + alpha * word_count as f64);
        self.negative_unseen_likelihood =
            alpha as f64 / (positive_word_count as f64 + alpha * word_count as f64);

        Ok(())
    }

    pub fn parse_csv(
        &self,
        path: &str,
        content_column: &str,
        label_column: &str,
    ) -> Result<Corpus, Box<dyn Error>> {
        let mut reader = Reader::from_path(path)?;
        let headers = reader
            .headers()
            .expect("Failed to read headers of CSV file");

        let column_present = |s: &str| headers.iter().any(|c| c == s);
        if !column_present(content_column) || !column_present(label_column) {
            return Err(Box::from("Columns are missing from CSV file"));
        }

        let find_column_position = |s: &str| headers.iter().position(|c| c == s);
        let content_column_position = find_column_position(content_column).unwrap();
        let label_column_position = find_column_position(label_column).unwrap();

        let mut corpus = Corpus {
            content: Vec::new(),
            labels: Vec::new(),
        };

        for result in reader.records() {
            let record = result.unwrap();
            let content = &record[content_column_position];
            let label = &record[label_column_position];
            corpus.content.push(content.to_string());
            corpus.labels.push(label.to_string());
        }

        Ok(corpus)
    }

    pub fn predict(&self, text: &str) -> bool {
        let tokens = Self::tokenise(text);
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

        // Normalisation using log-sum-exp
        let max = if positive_probability > negative_probability {
            positive_probability
        } else {
            negative_probability
        };

        // Use log-sum-exp to calculate sum of the logarithms of our posteriors
        let log_probability_sum =
            max + ((positive_probability - max).exp() + (negative_probability - max).exp()).ln();

        let positive_probability_normalised = (positive_probability - log_probability_sum).exp();
        let negative_probability_normalised = (negative_probability - log_probability_sum).exp();

        // println!(
        //     "Positive {}  negative {} ",
        //     positive_probability_normalised, negative_probability_normalised
        // );

        positive_probability_normalised > negative_probability_normalised
    }

    fn tokenise(text: &str) -> Vec<&str> {
        let tokens: Vec<&str> = TOKEN_REGEX
            .find_iter(text)
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        return tokens;
    }
}
