use polars::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::LazyLock;

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

impl Classifier {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha,
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
            .collect()
            .expect("Failed to count positive and negative records");

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
                    .extract_all(lit(r"\S+"))
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
            .filter((col("positive_counts") + col("negative_counts")).gt_eq(lit(5.0)))
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

        let v = unique_word_count;
        let alpha = self.alpha;
        let likelihoods = per_word_counts
            .clone()
            .lazy()
            .select([
                col("tokens"),
                ((col("positive_counts") + lit(alpha)) / lit(total_positive_words + alpha * v))
                    .alias("positive_likelihoods"),
                ((col("negative_counts") + lit(alpha)) / lit(total_negative_words + alpha * v))
                    .alias("negative_likelihoods"),
            ])
            .collect()?;

        let words = likelihoods.column("tokens")?.str()?;
        let pos = likelihoods.column("positive_likelihoods")?.f64()?;
        let neg = likelihoods.column("negative_likelihoods")?.f64()?;

        for ((token, p), n) in words.iter().zip(pos.iter()).zip(neg.iter()) {
            if let (Some(token), Some(p), Some(n)) = (token, p, n) {
                self.positive_likelihoods.insert(token.to_string(), p);
                self.negative_likelihoods.insert(token.to_string(), n);
            }
        }

        self.positive_unseen_likelihood =
            self.alpha / (total_positive_words + self.alpha * unique_word_count);
        self.negative_unseen_likelihood =
            self.alpha / (total_negative_words + self.alpha * unique_word_count);
        Ok(())
    }

    pub fn predict(&self, text: &str) -> bool {
        let tokens = Self::preprocess_text(text).split_whitespace().into_vec();
        let mut positive_probability = self.positive_probability.ln();
        let mut negative_probability = (1.0 - self.positive_probability).ln();
        for token in tokens {
            let token = token.as_str();
            let lp = self.positive_likelihoods.get(token);
            let ln = self.negative_likelihoods.get(token);

            // Skip words that appear in neither class
            if ln.is_none() && lp.is_none() {
                continue;
            }

            let likelihood_positive = lp.unwrap_or(&self.positive_unseen_likelihood);
            let likelihood_negative = ln.unwrap_or(&self.negative_unseen_likelihood);

            println!("{} {} ", likelihood_positive, likelihood_negative);

            positive_probability += likelihood_positive.ln();
            negative_probability += likelihood_negative.ln();
        }

        // Normalisation via logistic sigmoid
        let p = 1.0 / (1.0 + (negative_probability - positive_probability).exp());
        p > 0.995 // TODO: lwk just kind of guessed the threshold value should find this value via cross validation but idgaf
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
        // Remove stop words
        let processed = Self::tokenise(text)
            .into_iter()
            .map(|w| w.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // Remove all other characters not in the KEEP_REGEX
        // KEEP_REGEX also keeps emails and emoticons that otherwise would have been stripped
        KEEP_REGEX
            .replace_all(&processed, |caps: &regex::Captures| {
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
