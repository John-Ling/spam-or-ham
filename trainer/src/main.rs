mod classifier;
use classifier::Classifier;
use csv::Reader;
use std::{error::Error, path::PathBuf};
use tempfile::NamedTempFile;

fn main() -> Result<(), Box<dyn Error>> {
    // let _ = nuance_test(1.0, false);
    // let _ = cross_validation(1.0);

    // let _ = train_and_save();
    let _ = nuance_test(1.0, true);
    // let _ = single_test(true);

    Ok(())
}

fn train_and_save() -> Result<(), Box<dyn Error>> {
    let content_column = "body";
    let label_column = "label";
    let positive_label = "1";
    let negative_label = "0";
    let mut classifier = Classifier::new(1.0);
    let corpus = classifier.parse_csv(
        "trainer/src/data/Ling.csv",
        content_column,
        label_column,
        positive_label,
        negative_label,
    )?;
    println!("Training model");
    let _ = classifier.train(corpus)?;
    println!("Training completed");
    let _ = classifier.save_model("model.json")?;
    Ok(())
}

fn find_ideal_alpha() {
    let candidates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0];
    for i in 0..candidates.len() {
        let alpha = *candidates.get(i).unwrap();
        let _ = cross_validation(alpha);
    }
}

fn find_ideal_alpha_nuance_test() {
    let candidates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0];
    for i in 0..candidates.len() {
        let alpha = *candidates.get(i).unwrap();
        println!("TESTING α = {}", alpha);
        let _ = nuance_test(alpha, false);
        println!("");
    }
}

fn nuance_test(alpha: f64, pretrained: bool) -> Result<(), Box<dyn Error>> {
    let mut classifier = Classifier::new(alpha);
    let corpus = classifier.parse_csv("trainer/src/data/Ling.csv", "body", "label", "1", "0")?;
    if pretrained {
        let _ = classifier.load_pretrained("model.json");
    } else {
        let _ = classifier.train(corpus).expect("Failed training");
    }

    let spam_messages = [
        "CONGRATULATIONS!!! You are the LUCKY WINNER of $1,000,000 CASH!!! CLICK HERE NOW!!! 100% FREE!!! NO CATCH!!! LIMITED TIME ONLY!!! ACT NOW!!! BUY NOW!!!",
        "Dear Beloved Friend, I am a lawyer representing the late Mr. Johnson, who died with no heir. I need your help to transfer the funds. Reply with your full name, bank account number and phone number. Strictly confidential.",
        "Earn money fast with no experience and no investment required. Thousands of people are already getting rich. Sign up now and start earning cash today. Guaranteed income!",
        "Buy discount pills online with fast discreet shipping. Best prices guaranteed. Order now and save up to 80%. Click here to buy. No doctor visit required!",
        "You have been chosen as a winner in our customer loyalty draw. To claim your prize, simply verify your identity and provide your bank details. Act now, offer ends soon!",
        "Your premium membership is about to expire. Renew now to keep your benefits and receive a special discount. Click here to renew before it is too late.",
        "Dear business owner, join our private webinar and discover the secret strategies top companies use to multiply their revenue. Seats are limited, so register today and unlock your potential.",
        "We attempted to deliver your parcel today but no one was home. Please confirm your address and pay a small redelivery fee of $1.99 using the link below to reschedule.",
        "Hello, I'm not sure if you received my last email regarding the outstanding invoice. Could you open the attached document and confirm the payment details are correct? Regards, Accounts Team",
        "Hi, we noticed some unusual activity on your account. Please review your details at your earliest convenience to avoid any interruption to your service. Thanks, Customer Support",
    ];
    println!("RUNNING SPAM NUANCE TEST");
    let mut outcome: bool;
    let mut correct = 0;
    for i in 0..spam_messages.len() {
        print!("MESSAGE {}: ", i + 1);
        outcome = classifier.predict(spam_messages.get(i).unwrap());
        println!("{}", if outcome { "PASSED" } else { "FAILED" });
        if outcome {
            correct += 1;
        }
    }
    println!("{}/{} PASSED", correct, spam_messages.len());

    println!("RUNNING NON-SPAM NUANCE TEST");
    let non_spam_messages = [
            "Want to grab lunch at the new place around the corner? I was thinking 12:30. Let me know and I will book a table (^_^)",
            "Hey everyone, I uploaded my notes from today's lecture on conditional probability and Bayes' theorem. The practice problems on page 4 are tricky, so let's go through them on Thursday.",
            "Hi love, are you free for dinner on Sunday? Dad is making his famous lasagna and the kids would love to see you. Let me know if you can bring dessert. Mum x",
            "Jordan requested your review on “Fix tokenizer edge case”. The changes add handling for empty strings and include two new unit tests. Please take a look when you have a moment.",
            "Dear Julie,

    Thank you for taking the time to sort out handover details and for the opportunity for us to demo the project's latest build.

    The handover document is attached. Email us if you have any questions or suggestions for changes.

    After reading if you're confident with proceeding to the handover, provide a date and an email associated with a GitHub account and we'll happily transfer the repository over as well as necessary secrets as outlined.

    Many thanks,
    Campbell Jackson. ",
            "Hi Julie, hope you are doing well.

    I'm Peter, the Scrum Master of Team Google3, working on MediMate. Campbell, our Product Owner, usually handles our communication with you, but I'm stepping in to help organise the handover.

    We're nearly finished building MediMate and would like to hand it over to Medical Pantry.

    We suggest doing the handover by email, with:

        A single handover document
        A recorded video demo of the app
    Could you let us know by 04/11/2026? Our deadline is 12 November, so we'd like to start as soon as we can. Any passwords or keys will be sent to you separately.

    Thanks so much for working with us on this.

    Kind regards,

    Peter Huang
    Scrum Master, Team Google3

    COMP30022 IT Project, University of Melbourne",
    "Hi Peter,

    Thanks for the detailed message.

    Let's schedule a meeting as well as have the handover by email.

    Do you have time tomorrow at 12pm for the handover?

    Thank you, ",
            "Hi team, the testing phase is complete and we found three bugs, all now fixed. Next week we will start the deployment checklist. Please review the document and send comments by Wednesday.",
            "Hey, something has come up and I need to reschedule our 2pm catch-up. Would Friday morning work instead? Happy to do a quick call if that is easier. Thanks! :)",
            "Hi Priya, please find the invoice for September attached. Payment is due within 14 days. Let me know if you need any changes to the details. Cheers, Marcus",
            "Thanks for your recent purchase! As a returning customer, you get free shipping on orders over $50 for the next 30 days. This offer applies to your account only and expires on 30 November.",
            "Your October account statement is now available. Please log in to your online banking portal to view it. If you have any questions about a transaction, call the number on the back of your card.",
            "Hi Sam, good news: your order #48213 has shipped and should arrive within 3-5 business days. You can track your parcel using the tracking number below. Thanks for shopping with us!",
        ];

    correct = 0;
    for i in 0..non_spam_messages.len() {
        print!("MESSAGE {}: ", i + 1);
        outcome = classifier.predict(non_spam_messages.get(i).unwrap());
        println!("{}", if !outcome { "PASSED" } else { "FAILED" });
        if !outcome {
            correct += 1;
        }
    }
    println!("{}/{} PASSED", correct, non_spam_messages.len());
    Ok(())
}

fn single_test(pretrained: bool) -> Result<(), Box<dyn Error>> {
    let content_column = "body";
    let label_column = "label";
    let positive_label = "1";
    let negative_label = "0";

    let mut classifier = Classifier::new(0.55);
    if pretrained {
        let _ = classifier.load_pretrained("model.json")?;
    } else {
        let corpus = classifier.parse_csv(
            "trainer/src/data/Ling.csv",
            content_column,
            label_column,
            positive_label,
            negative_label,
        )?;
        let _ = classifier.train(corpus);
    }

    let mut false_positive_count = 0.0;
    let mut false_negative_count = 0.0;
    let mut true_positive_count = 0.0;

    let mut reader = Reader::from_path("trainer/src/data/Ling.csv")?;
    let headers = reader.headers()?;
    let find_column_position = |s: &str| headers.iter().position(|c| c == s);
    let content_column_position = find_column_position(content_column).unwrap();
    let label_column_position = find_column_position(label_column).unwrap();
    println!("Calculating metrics");
    for result in reader.records() {
        let record = result?;
        let content = &record[content_column_position];
        let is_positive = classifier.predict(content);
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
    println!(
        "PRECISION: {}\nRECALL:    {}\nF1:        {}",
        precision, recall, f1
    );
    Ok(())
}

fn split_csv(k: usize) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut reader = Reader::from_path("trainer/src/data/Ling.csv")?;
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

fn cross_validation(alpha: f64) -> Result<(), Box<dyn Error>> {
    let content_column = "body";
    let label_column = "label";
    let positive_label = "1";
    let negative_label = "0";

    let k: usize = 10;
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

        let mut fold_classifier = Classifier::new(alpha);
        let corpus = fold_classifier.parse_csv(
            train_path.to_str().unwrap(),
            content_column,
            label_column,
            positive_label,
            negative_label,
        );
        let _ = fold_classifier.train(corpus.unwrap());

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
        println!(
            "PRECISION: {}\nRECALL:    {}\nF1:        {}",
            precision, recall, f1
        );
        average_precision += precision;
        average_recall += recall;
        average_f1 += f1;
    }

    average_precision /= k as f64;
    average_recall /= k as f64;
    average_f1 /= k as f64;
    println!("FINAL RESULTS");
    println!(
        "PRECISION: {}\nRECALL:    {}\nF1:        {}",
        average_precision, average_recall, average_f1
    );
    Ok(())
}

fn merge_csv_chunks(
    paths: &[PathBuf],
    exclude: usize,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
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
