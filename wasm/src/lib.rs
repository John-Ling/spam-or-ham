use wasm_bindgen::prelude::*;
mod classifier;
use classifier::Classifier;
use std::sync::OnceLock;

static CLASSIFIER_INSTANCE: OnceLock<Classifier> = OnceLock::new();

#[wasm_bindgen]
pub fn load_classifier(model_json_string: &str) -> bool {
    let mut classifier = Classifier::new();

    let successful = classifier.load_pretrained(model_json_string).is_ok();
    CLASSIFIER_INSTANCE.set(classifier).ok();
    return successful;
}

#[wasm_bindgen]
pub fn predict(string: &str) -> bool {
    web_sys::console::log_2(&"{}".into(), &JsValue::from_str(string));
    let classifier = CLASSIFIER_INSTANCE.get().unwrap();
    return classifier.predict(string);
}
