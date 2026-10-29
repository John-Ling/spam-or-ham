use wasm_bindgen::prelude::*;
mod classifier;
use classifier::Classifier;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    return String::from(name);
}

#[wasm_bindgen]
pub fn load_classifier(model_json_string: &str) -> String {
    let mut classifier = Classifier::new();
    let _ = classifier.load_pretrained_from_string(model_json_string);
    let result = classifier.predict(
        "Good morning John

Could you please reply to this email with your personal cell phone number. 

Kind Regards,
Eric Zhang",
    );

    let result = if result { "spam" } else { "ham" };
    return String::from(result);
}
