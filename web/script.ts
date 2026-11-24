import init, { load_classifier, predict } from "./public/wasm.js"
import { SAMPLE_TEXTS } from "./samples.js";
import { generateOutputHTML, getCaretOffset, moveCaretToOffset, updateProbabilityCounterUI } from "./utils.js";

let loaded = false
let inputField: HTMLDivElement | null = null
let tooltip: HTMLDivElement | null = null
let previousOutcome: boolean | null = null

const clear = new Audio("/clear.wav")
const click = new Audio("/click.wav")

document.addEventListener("DOMContentLoaded", async () => {
  inputField = document.getElementById("input-field") as HTMLDivElement
  tooltip = document.createElement("div")
  tooltip.id = "tooltip"
  document.body.append(tooltip)

  let spamHeader = document.getElementById("spam-header")!;
  let hamHeader = document.getElementById("ham-header")!;

  document.getElementById("input-field")?.addEventListener("input", (e) => {
    let inputField = e.target as HTMLDivElement
    let content = inputField.innerText

    if (content.trim() === "") {
      spamHeader.classList = ""
      hamHeader.classList = ""
      inputField.innerHTML = ""
      previousOutcome = null
      clear.currentTime = 0
      clear.play()

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

    if (previousOutcome === null || isSpam !== previousOutcome) {
      click.currentTime = 0
      click.play()
    }

    previousOutcome = isSpam
    spamHeader.classList = isSpam ? "lift" : "sink"
    hamHeader.classList = isSpam ? "sink" : "lift"

    const caretOffset = getCaretOffset(inputField)
    const output = generateOutputHTML(content, positiveLikelihoods, negativeLikelihoods)
    inputField.innerHTML = output;
    moveCaretToOffset(inputField, caretOffset)
    updateProbabilityCounterUI(positiveProbability, negativeProbability)
  })

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

  const sampleButtonContainer = document.getElementById("sample-button-container")
  if (sampleButtonContainer) {
    Object.keys(SAMPLE_TEXTS).forEach((key, i) => {
      const button = document.createElement("button");

      button.textContent = `Sample ${i + 1}`
      button.dataset.key = key
      button.addEventListener("click", e => {
        e.preventDefault()
        loadSampleIntoInput(key)
      })
      sampleButtonContainer.append(button)
    })
  }

  document.getElementById("clear-button")?.addEventListener("click", () => {
    if (!inputField) {
      return
    }
    inputField.innerText = ""
    inputField.dispatchEvent(new Event("input", { bubbles: true }))
  })


  await init();
  let res = await fetch("/model.json");
  loaded = load_classifier(await res.text())
})

const loadSampleIntoInput = (key: keyof typeof SAMPLE_TEXTS): void => {
  if (!inputField) {
    return
  }

  inputField.innerText = SAMPLE_TEXTS[key] ?? ""
  inputField.dispatchEvent(new Event("input", { bubbles: true }))
}

const hideTooltip = () => {
  if (tooltip) {
    tooltip.style.display = 'none';
  }
};
