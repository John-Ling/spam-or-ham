use polars::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::LazyLock;
use stop_words::{Language, get};

pub struct Corpus {
    content: Vec<String>,
    labels: Vec<String>,
}

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

const TRAILING_PUNCTUATION_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\w+|[^\w\s]").unwrap());
const EMAIL_REGEX: &str = r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*";
const EMOTICON_REGEX: &str = r"[:;=8]-?[)(DPOpo|\\/*]|\([a-z]{2,10}\)";

static KEEP_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    {
        Regex::new(
            format!(
                r"({})|({})|([a-zA-Z0-9\s!?$%'])|.",
                EMAIL_REGEX, EMOTICON_REGEX
            )
            .as_str(),
        )
    }
    .unwrap()
});

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

    pub fn save_model(&mut self, model_path: &str) -> Result<(), Box<dyn Error>> {
        let serialised = Classifier {
            alpha: self.alpha,
            content_column: self.content_column.to_string(),
            label_column: self.label_column.to_string(),
            negative_label: self.negative_label.to_string(),
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

    pub fn load_pretrained(&mut self, model_path: &str) -> Result<(), Box<dyn Error>> {
        let read_file = File::open(model_path)?;
        let reader = BufReader::new(read_file);
        let model: Classifier = serde_json::from_reader(reader)?;
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

    pub fn load_pretrained_from_string(
        &mut self,
        model_string: &str,
    ) -> Result<(), Box<dyn Error>> {
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

    pub fn train(&mut self, corpus: DataFrame) -> Result<(), Box<dyn Error>> {
        let processed = self.preprocess_corpus(corpus)?;
        let label_column = &self.label_column;
        let content_column = &self.content_column;
        let positive_label = &self.positive_label;
        let negative_label = &self.negative_label;

        let string_typed_label = || col(label_column).cast(DataType::String);

        // Get total count of positive and negative records
        let positive_negative_record_counts = processed
            .clone()
            .lazy()
            .select([
                string_typed_label()
                    .filter(string_typed_label().eq(lit(positive_label.as_str())))
                    .len()
                    .alias("positive_count"),
                string_typed_label()
                    .filter(string_typed_label().eq(lit(negative_label.as_str())))
                    .len()
                    .alias("negative_count"),
            ])
            .collect()?;

        let total_positive_records = positive_negative_record_counts
            .column("positive_count")?
            .get(0)?
            .try_extract::<f64>()?;

        let total_negative_records = positive_negative_record_counts
            .column("negative_count")?
            .get(0)?
            .try_extract::<f64>()?;

        let total_record_count = total_positive_records + total_negative_records;
        self.positive_probability = total_positive_records / total_record_count;

        // Calculate frequency of words in positive and negative records
        let per_word_counts = processed
            .lazy()
            .select([
                col(label_column).cast(DataType::String),
                col(content_column)
                    .str()
                    .extract_all(lit(r"\w+|[^\w\s]"))
                    .alias("tokens"),
            ])
            .explode(
                by_name(["tokens"], true, true),
                ExplodeOptions {
                    empty_as_null: true,
                    keep_nulls: true,
                },
            )
            .group_by([col("tokens")])
            .agg([
                col(label_column)
                    .eq(lit(positive_label.as_str()))
                    .sum()
                    .cast(DataType::Float64)
                    .alias("positive_counts"),
                col(label_column)
                    .eq(lit(negative_label.as_str()))
                    .sum()
                    .cast(DataType::Float64)
                    .alias("negative_counts"),
            ])
            .collect()?;

        let unique_word_count = per_word_counts.column("tokens")?.len() as f64;
        let total_positive_words = per_word_counts
            .column("positive_counts")?
            .f64()?
            .sum()
            .unwrap();
        let total_negative_words = per_word_counts
            .column("negative_counts")?
            .f64()?
            .sum()
            .unwrap();

        // Calculate likelihoods
        let positive_counts_per_word = per_word_counts.column("positive_counts")?.f64()?;
        let negative_counts_per_word = per_word_counts.column("negative_counts")?.f64()?;
        let words = per_word_counts.column("tokens")?.str()?;
        for ((token, positive_count), negative_count) in words
            .iter()
            .zip(positive_counts_per_word.iter())
            .zip(negative_counts_per_word.iter())
        {
            if let (Some(token), Some(positive_count), Some(negative_count)) =
                (token, positive_count, negative_count)
            {
                self.positive_likelihoods.insert(
                    token.to_string(),
                    (positive_count + self.alpha)
                        / (total_positive_words + self.alpha * unique_word_count),
                );
                self.negative_likelihoods.insert(
                    token.to_string(),
                    (negative_count + self.alpha)
                        / (total_negative_words + self.alpha * unique_word_count),
                );
            }
        }

        // let mut sorted: Vec<(&String, &f64)> = self.positive_likelihoods.iter().collect();
        // sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

        // println!("{:#?}", sorted);

        self.positive_unseen_likelihood =
            self.alpha / (total_positive_words + self.alpha * unique_word_count);
        self.negative_unseen_likelihood =
            self.alpha / (total_negative_words + self.alpha * unique_word_count);
        Ok(())
    }

    fn preprocess_corpus(&self, corpus: DataFrame) -> Result<DataFrame, Box<dyn Error>> {
        let content_column = self.content_column.as_str();
        let processed_corpus = corpus.lazy().with_columns([col(content_column).map(
            move |c| {
                let ca = c.str()?;
                let out: StringChunked = ca.apply_values(|t| Self::preprocess_text(t).into());

                Ok(out.into_column())
            },
            |_schema, field| Ok(field.clone()),
        )]);

        Ok(processed_corpus.collect()?)
    }

    fn preprocess_text(text: &str) -> String {
        let stop_words: HashSet<&str> = get(Language::English).iter().map(|s| *s).collect();

        // Remove stop words
        let without_stop_words = text
            .split_whitespace()
            .filter(|w| !stop_words.contains(w.to_lowercase().as_str()))
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

    pub fn parse_csv(
        &mut self,
        path: &str,
        content_column: &str,
        label_column: &str,
        positive_label: &str,
        negative_label: &str,
    ) -> Result<DataFrame, Box<dyn Error>> {
        let mut df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?;
        df = df.drop_nulls(Some(&[content_column]))?;
        self.content_column = content_column.to_string();
        self.label_column = label_column.to_string();
        self.positive_label = positive_label.to_string();
        self.negative_label = negative_label.to_string();
        Ok(df)
    }

    pub fn predict(&self, text: &str) -> bool {
        let processed = Self::preprocess_text(text);
        let tokens = Self::tokenise(&processed);
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
        positive_probability_normalised > negative_probability_normalised
    }

    // Split text into tokens including stripping trailing punctuation as their own token
    // i.e hello! becomes "hello", "!"
    fn tokenise(text: &str) -> Vec<&str> {
        let tokens: Vec<&str> = TRAILING_PUNCTUATION_REGEX
            .find_iter(text)
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();
        return tokens;
        // let bigrams = tokens.windows(2).map(|w| (w[0], w[1])).collect();
        // return bigrams;
    }
}
