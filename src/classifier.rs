use std::{collections::HashMap, fs::read_to_string};

pub struct Corpus {
    content: Vec<String>,
    labels: Vec<String>,
}

pub struct Classifier {
    weights: HashMap<String, f32>,
}

impl Classifier {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    fn train(corpus: Corpus) {
        // let mut spam_record_count = 0;
        // let mut record_count = 0;
        // let mut word_frequencies: HashMap<String, usize> = HashMap::new();

        // for result in reader.records() {
        //     let record = result.unwrap();
        //     let label = &record[label_column_position];
        //     if label == "1" {
        //         spam_record_count += 1;
        //         let content = &record[content_column_position];
        //         let words = content.split_whitespace().into_iter().map(|w| {
        //             word_frequencies
        //                 .entry(w)
        //                 .and_modify(|f| *f += 1)
        //                 .or_insert(1)
        //         });
        //     }

        //     record_count += 1;
        // }

        // let spam_probability: f32 = spam_record_count as f32 / record_count as f32;
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
