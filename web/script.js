import init, { load_classifier, predict } from "./public/wasm.js";
import { updateTokenProbabilitiesUI } from "./utils.js";
let loaded = false;
document.addEventListener("DOMContentLoaded", async () => {
    await init();
    let res = await fetch("/model.json");
    loaded = load_classifier(await res.text());
});
let spamHeader = document.getElementById("spam-header");
let hamHeader = document.getElementById("ham-header");
document.getElementById("input-field")?.addEventListener("input", (e) => {
    let inputField = e.target;
    let content = inputField.innerText;
    if (content.trim() === "") {
        spamHeader.classList = "";
        hamHeader.classList = "";
        inputField.innerHTML = "";
        return;
    }
    let { outcome: isSpam, positive_probabilities, negative_probabilities } = predict(content);
    spamHeader.classList = isSpam ? "lift" : "sink";
    hamHeader.classList = isSpam ? "sink" : "lift";
    updateTokenProbabilitiesUI({ content, inputField, positive_probabilities, negative_probabilities });
});
//# sourceMappingURL=script.js.map