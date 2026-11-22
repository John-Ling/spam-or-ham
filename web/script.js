import init, { load_classifier, predict } from "./public/wasm.js";
import { updateTokenProbabilitiesUI } from "./utils.js";
let loaded = false;
let inputField = null;
let tooltip = null;
document.addEventListener("DOMContentLoaded", async () => {
    await init();
    let res = await fetch("/model.json");
    loaded = load_classifier(await res.text());
    inputField = document.getElementById("input-field");
    tooltip = document.createElement("div");
    tooltip.id = "tooltip";
    document.body.append(tooltip);
    inputField?.addEventListener("mouseover", (e) => {
        const selectedToken = e.target.closest(".has-tooltip");
        if (!selectedToken || !tooltip) {
            return;
        }
        tooltip.textContent = selectedToken.dataset.tip ?? "";
        tooltip.style.display = "block";
        const tokenBoundingRectangle = selectedToken.getBoundingClientRect();
        const width = tooltip.offsetWidth;
        const height = tooltip.offsetHeight;
        // Centre above the token, clamp to viewport, flip below if no room
        let left = tokenBoundingRectangle.left + tokenBoundingRectangle.width / 2 - width / 2;
        left = Math.max(4, Math.min(left, window.innerWidth - width - 4));
        let top = tokenBoundingRectangle.top - height - 6;
        if (top < 4)
            top = tokenBoundingRectangle.bottom + 6;
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
    });
    inputField?.addEventListener('mouseout', (e) => {
        const to = e.relatedTarget;
        if (!to?.closest('.has-tooltip')) {
            hideTooltip();
        }
    });
    inputField?.addEventListener("scroll", hideTooltip);
    window.addEventListener("scroll", hideTooltip, true);
});
const hideTooltip = () => {
    if (tooltip) {
        tooltip.style.display = 'none';
    }
};
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