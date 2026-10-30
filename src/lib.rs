use wasm_bindgen::prelude::*;
mod classifier;
use classifier::Classifier;
use std::sync::OnceLock;

static CLASSIFIER_INSTANCE: OnceLock<Classifier> = OnceLock::new();

#[wasm_bindgen]
pub fn load_classifier(model_json_string: &str) -> bool {
    let mut classifier = Classifier::new();

    let successful = classifier
        .load_pretrained_from_string(model_json_string)
        .is_ok();
    CLASSIFIER_INSTANCE.set(classifier).ok();
    return successful;
}

#[wasm_bindgen]
pub fn predict(string: &str) -> bool {
    web_sys::console::log_2(&"{}".into(), &JsValue::from_str(string));

    let classifier_loaded = CLASSIFIER_INSTANCE.get().is_none();

    web_sys::console::log_2(
        &"Classifier is {}".into(),
        &JsValue::from_bool(classifier_loaded),
    );
    let classifier = CLASSIFIER_INSTANCE.get().unwrap();
    return classifier.predict(string);
}
