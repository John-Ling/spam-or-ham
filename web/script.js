import init, { load_classifier, predict } from "./public/wasm.js";
let loaded = false;
init().then(async () => {
    let res = await fetch("/model.json");
    loaded = load_classifier(await res.text());
    console.log(loaded);
});
let spamHeader = document.getElementById("spam-header");
let hamHeader = document.getElementById("ham-header");
document.getElementById("input-field")?.addEventListener("input", (e) => {
    let inputContent = e.target.value;
    if (inputContent === "") {
        spamHeader.classList = "";
        hamHeader.classList = "";
        return;
    }
    let { outcome: isSpam, positive_probabilities, negative_probabilities } = predict(inputContent);
    spamHeader.classList = isSpam ? "lift" : "sink";
    hamHeader.classList = isSpam ? "sink" : "lift";
});
//# sourceMappingURL=script.js.map