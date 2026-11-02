import init, { load_classifier, predict } from "./public/pkg/spam_or_ham.js";
let loaded = false;
init().then(async () => {
    let res = await fetch("/model.json");
    let model_as_text = await res.text();
    console.log("model", model_as_text);
    loaded = load_classifier(model_as_text);
    let text = loaded ? "yay" : "nay";
    document.getElementById("name").innerText = text;
});
let spamHeader = document.getElementById("spam-header");
let hamHeader = document.getElementById("ham-header");
document.getElementById("input-field")?.addEventListener("input", (e) => {
    let inputContent = e.target.value;
    console.log("Entered ", inputContent);
    let isSpam = predict(inputContent);
    spamHeader.classList = isSpam ? "lift" : "sink";
    hamHeader.classList = isSpam ? "sink" : "lift";
});
//# sourceMappingURL=script.js.map