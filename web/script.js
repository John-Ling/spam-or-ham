import init, { load_classifier, predict } from "./public/wasm.js";
let loaded = false;
await init();
let res = await fetch("/model.json");
loaded = load_classifier(await res.text());
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
    updateTokenProbabilitiesUI(content, inputField, positive_probabilities, negative_probabilities);
});
const preprocessText = (text) => {
    const TRAILING_PUNCTUATION_REGEX = /(?:\r?\n)+|\w+|\p{P}+/gu;
    const EMAIL_REGEX = /[a-zA-Z.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*/;
    const EMOTICON_REGEX = /[:;=8]-?[)(DPOpo|\\/*]|\([a-z]{2,10}\)/;
    const TOKEN_REGEX = new RegExp(`(${EMAIL_REGEX.source})|(${EMOTICON_REGEX.source})|(${TRAILING_PUNCTUATION_REGEX.source})`, 'gu');
    const tokens = text.match(TOKEN_REGEX);
    if (!tokens) {
        return "";
    }
    return tokens.map(t => t.toLowerCase()).join(" ");
};
const updateTokenProbabilitiesUI = (content, inputField, positive_probabilities, negative_probabilities) => {
    const caretOffset = getCaretOffset(inputField);
    let t = preprocessText(content);
    const tokens = content.match(/\w+|[\p{P}\p{S}]+|\s+/gu) ?? [];
    let processed_tokens = t.split(/\s+/) ?? [];
    let output = "";
    let counter = 0;
    for (const token of tokens) {
        if (/\s+/.test(token)) {
            output += token;
            continue;
        }
        if (processed_tokens.includes(token.toLowerCase())) {
            console.log(positive_probabilities[counter]);
            console.log(negative_probabilities[counter]);
            output += `<button class="token has-tooltip highlight-yellow" data-tip="POSITIVE: ${parseFloat(positive_probabilities[counter].toFixed(6))} / NEGATIVE: ${parseFloat(negative_probabilities[counter].toFixed(6))}"
>${token}</button>`;
            counter += 1;
            continue;
        }
        output += `<button class="token highlight-red">${token}</button>`;
    }
    inputField.innerHTML = output;
    moveCaretToOffset(inputField, caretOffset);
};
const tokenise = (text) => {
    const TRAILING_PUNCTUATION_REGEX = /(?:\r?\n)+|\w+|\p{P}+/gu;
    return text.match(TRAILING_PUNCTUATION_REGEX) ?? [];
};
const getCaretOffset = (element) => {
    const selection = window.getSelection();
    if (!selection || !selection.rangeCount) {
        return 0;
    }
    const range = selection.getRangeAt(0).cloneRange();
    range.selectNodeContents(element);
    range.setEnd(selection.getRangeAt(0).endContainer, selection.getRangeAt(0).endOffset);
    return range.toString().length;
};
const moveCaretToOffset = (element, offset) => {
    const range = document.createRange();
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    let node, count = 0;
    while ((node = walker.nextNode())) {
        const next = count + node.length;
        if (offset <= next) {
            // Found where the caret needs to be moved to 
            range.setStart(node, offset - count);
            range.collapse(true);
            const selection = window.getSelection();
            selection?.removeAllRanges();
            selection?.addRange(range);
            return;
        }
        count = next;
    }
    // If the offset is past the end, place the caret at the end of the content
    range.selectNodeContents(element);
    range.collapse(false);
    const selection = window.getSelection();
    selection?.removeAllRanges();
    selection?.addRange(range);
};
//# sourceMappingURL=script.js.map