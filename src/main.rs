mod classifier;
use classifier::Classifier;
fn main() {
    let classifier = Classifier::new();
    let _ = classifier.parse_csv("src/data/Ling.csv", "body", "label");
}

fn train() {}

fn classify() {}
