use regex::Regex;
use std::collections::HashMap;

pub struct Corpus {
    content: Vec<String>,
    labels: Vec<String>,
}

pub struct Classifier {
    positive_label: String,
    positive_probability: f64,
    weights: HashMap<String, f64>,
}

impl Classifier {
    pub fn new(spam_label: &str) -> Self {
        Self {
            positive_label: spam_label.to_string(),
            positive_probability: 0.0,
            weights: HashMap::new(),
        }
    }

    pub fn train(mut self, corpus: Corpus) -> Result<(), String> {
        let mut positive_record_count = 0;
        let mut record_count = 0;
        let mut word_frequencies: HashMap<String, u32> = HashMap::new();

        if corpus.content.len() != corpus.labels.len() {
            return Err(String::from(
                "Number of labels in corpus does not match the amount of content",
            ));
        }

        let re = Regex::new(r"\w+|[^\w\s]").unwrap();
        for (_, (content, label)) in corpus.content.iter().zip(corpus.labels.iter()).enumerate() {
            if *label == self.positive_label {
                let tokens: Vec<&str> = re
                    .find_iter(content)
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>();

                positive_record_count += 1;
                for w in tokens {
                    word_frequencies
                        .entry(String::from(w))
                        .and_modify(|f| *f += 1)
                        .or_insert(2); // Insert 2 instead of 1 to implicitly add Laplace smoothing
                }
            }
            record_count += 1;
        }

        let positive_word_count: u32 = word_frequencies.values().sum();
        let positive_probability = positive_record_count as f64 / record_count as f64;
        self.positive_probability = positive_probability;

        let mut min = f64::MAX;
        let mut max = f64::MIN;

        // Compute conditional/posterior probabilities
        for (_, (word, frequency)) in word_frequencies.iter().enumerate() {
            let posterior = *frequency as f64 / positive_word_count as f64;
            if posterior < min {
                min = posterior;
            }

            if posterior > max {
                max = posterior;
            }

            self.weights.insert(
                word.to_string(),
                *frequency as f64 / positive_word_count as f64,
            );
        }

        // Normalise
        for word in word_frequencies.keys() {
            self.weights
                .entry(word.to_string())
                .and_modify(|p| *p = (*p - min) / (max - min));
        }

        let mut s: Vec<(&String, &f64)> = self.weights.iter().collect();
        s.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        println!("{:#?}", s);
        Ok(())
    }

    pub fn parse_csv(
        &self,
        path: &str,
        content_column: &str,
        label_column: &str,
    ) -> Result<Corpus, String> {
        let data = std::fs::read_to_string(path).expect("could not read data from file");
        let mut reader = csv::Reader::from_reader(data.as_bytes());
        let headers = reader
            .headers()
            .expect("Failed to read headers of CSV file");

        let column_present = |s: &str| headers.iter().any(|c| c == s);
        if !column_present(content_column) || !column_present(label_column) {
            return Err(String::from("Columns are missing from CSV file"));
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

    fn predict(text: String) {}
}
