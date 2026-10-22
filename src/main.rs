mod classifier;
use classifier::Classifier;
fn main() {
    let classifier = Classifier::new("1");
    let corpus = classifier.parse_csv("src/data/ling_cleaned.csv", "body", "label");
    let _ = classifier.train(corpus.unwrap());
}

fn train() {}

fn classify() {}
