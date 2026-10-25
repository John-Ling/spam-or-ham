mod classifier;
use std::{error::Error, path::PathBuf};

use classifier::Classifier;
use csv::Reader;
use tempfile::NamedTempFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let content_column = "body";
    let label_column = "label";
    let positive_label = "1";
    let k = 10;
    let paths = split_csv(k).unwrap();

    let mut average_f1 = 0.0;
    let mut average_precision = 0.0;
    let mut average_recall = 0.0;

    for i in 0..k {
        if paths[i].to_str().is_none() {
            println!("Could not read path during training");
            continue;
        }
        let test_path = paths[i].to_str().unwrap();

        // Train on the other k - 1 partitions, merged into a single CSV
        let (train_file, train_path) = NamedTempFile::new()?.keep()?;
        drop(train_file);
        merge_csv_chunks(&paths, i, train_path.to_str().unwrap())?;

        let mut fold_classifier = Classifier::new();
        let corpus = fold_classifier.parse_csv(
            train_path.to_str().unwrap(),
            content_column,
            label_column,
        );
        let _ = fold_classifier.train(corpus.unwrap(), "1");

        let mut false_positive_count: f64 = 0.0;
        let mut false_negative_count: f64 = 0.0;
        let mut true_positive_count: f64 = 0.0;

        // Test on the isolated ith partition
        let mut reader = Reader::from_path(test_path)?;
        let headers = reader.headers().unwrap();
        let find_column_position = |s: &str| headers.iter().position(|c| c == s);
        let content_column_position = find_column_position(content_column).unwrap();
        let label_column_position = find_column_position(label_column).unwrap();

        for result in reader.records() {
            let record = result.unwrap();
            let content = &record[content_column_position];
            let is_positive = fold_classifier.predict(content);
            let actual = if &record[label_column_position] == positive_label {
                true
            } else {
                false
            };

            if is_positive && actual {
                true_positive_count += 1.0;
            } else if is_positive && !actual {
                false_positive_count += 1.0;
            } else if !is_positive && actual {
                false_negative_count += 1.0;
            }
        }

        let precision = true_positive_count / (true_positive_count + false_positive_count);
        let recall = true_positive_count / (true_positive_count + false_negative_count);
        let f1 = 2.0 * precision * recall / (precision + recall);

        println!("TEST RUN {} RESULTS", i);
        println!("PRECISION: {}\nRECALL: {}\nF1: {}", precision, recall, f1);
        average_precision += precision;
        average_recall += recall;
        average_f1 += f1;
    }

    average_precision /= k as f64;
    average_recall /= k as f64;
    average_f1 /= k as f64;
    println!("FINAL RESULTS");
    println!(
        "PRECISION: {}\nRECALL: {}\nF1: {}",
        average_precision, average_recall, average_f1
    );
    Ok(())
}

fn split_csv(k: usize) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut reader = Reader::from_path("src/data/ling_cleaned.csv")?;
    let headers = reader.headers()?.clone();
    let records: Vec<csv::StringRecord> = reader.records().collect::<Result<_, _>>()?;
    let n = records.len();
    let base = n / k;
    let extra = n % k;

    let mut paths = Vec::with_capacity(k);
    let mut start = 0;

    // Split csv into K parts and save to temporary files
    for i in 0..k {
        let len = base + if i < extra { 1 } else { 0 };
        let chunk = &records[start..start + len];
        start += len;

        // keep() stops the file being deleted when the handle is dropped
        let (file, path) = NamedTempFile::new()?.keep()?;
        drop(file);

        let mut writer = csv::WriterBuilder::new().from_path(&path)?;
        writer.write_record(&headers)?;
        for rec in chunk {
            writer.write_record(rec)?;
        }
        writer.flush()?;

        paths.push(path);
    }
    Ok(paths)
}

fn merge_csv_chunks(paths: &[PathBuf], exclude: usize, out_path: &str) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::WriterBuilder::new().from_path(out_path)?;
    let mut header_written = false;

    for (j, path) in paths.iter().enumerate() {
        if j == exclude {
            continue;
        }

        let mut reader = Reader::from_path(path)?;
        if !header_written {
            writer.write_record(reader.headers()?)?;
            header_written = true;
        } else {
            // Skip headers after the first chunk
            let _ = reader.headers()?;
        }

        for result in reader.records() {
            let record = result?;
            writer.write_record(&record)?;
        }
    }

    writer.flush()?;
    Ok(())
}
