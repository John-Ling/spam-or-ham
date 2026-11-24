import init, { load_classifier, predict } from "./public/wasm.js"
import { generateOutputHTML, getCaretOffset, moveCaretToOffset, updateProbabilityCounterUI } from "./utils.js";

let loaded = false

let inputField = null
let tooltip: HTMLDivElement | null = null
document.addEventListener("DOMContentLoaded", async () => {
  await init();
  let res = await fetch("/model.json");
  loaded = load_classifier(await res.text())
  inputField = document.getElementById("input-field")
  tooltip = document.createElement("div")
  tooltip.id = "tooltip"
  document.body.append(tooltip)

  inputField?.addEventListener("mouseover", (e) => {
    const selectedToken = (e.target as HTMLElement).closest<HTMLSpanElement>(".has-tooltip")
    if (!selectedToken || !tooltip) {
      return
    }

    tooltip.textContent = selectedToken.dataset.tip ?? ""
    tooltip.style.display = "block"
    const tokenBoundingRectangle = selectedToken.getBoundingClientRect()
    const width = tooltip.offsetWidth
    const height = tooltip.offsetHeight

    // Centre above the token, clamp to viewport, flip below if no room
    let left = tokenBoundingRectangle.left + tokenBoundingRectangle.width / 2 - width / 2;
    left = Math.max(4, Math.min(left, window.innerWidth - width - 4));
    let top = tokenBoundingRectangle.top - height - 6;
    if (top < 4) {
      top = tokenBoundingRectangle.bottom + 6;
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  })

  inputField?.addEventListener('mouseout', (e) => {
    const to = (e as MouseEvent).relatedTarget as HTMLElement | null;
    if (!to?.closest('.has-tooltip')) {
      hideTooltip();
    }
  });

  inputField?.addEventListener("scroll", hideTooltip)
  window.addEventListener("scroll", hideTooltip, true)

})

const hideTooltip = () => {
  if (tooltip) {
    tooltip.style.display = 'none';
  }
};


let spamHeader = document.getElementById("spam-header")!;
let hamHeader = document.getElementById("ham-header")!;

document.getElementById("input-field")?.addEventListener("input", (e) => {
  let inputField = e.target as HTMLDivElement
  let content = inputField.innerText

  if (content.trim() === "") {
    spamHeader.classList = ""
    hamHeader.classList = ""
    inputField.innerHTML = ""
    updateProbabilityCounterUI(0.0, 0.0)
    return
  }

  let {
    outcome: isSpam,
    positive_likelihoods: positiveLikelihoods,
    negative_likelihoods: negativeLikelihoods,
    positive_probability: positiveProbability,
    negative_probability: negativeProbability
  } = predict(content)
  spamHeader.classList = isSpam ? "lift" : "sink"
  hamHeader.classList = isSpam ? "sink" : "lift"
  const caretOffset = getCaretOffset(inputField)
  const output = generateOutputHTML(content, positiveLikelihoods, negativeLikelihoods)
  inputField.innerHTML = output;
  moveCaretToOffset(inputField, caretOffset)
  updateProbabilityCounterUI(positiveProbability, negativeProbability)
})

