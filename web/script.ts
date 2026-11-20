import init, { load_classifier, predict } from "./public/wasm.js"

let loaded = false
let inputFieldBuffer = ""

await init();
let res = await fetch("/model.json");
loaded = load_classifier(await res.text())

let spamHeader = document.getElementById("spam-header")!;
let hamHeader = document.getElementById("ham-header")!;

document.getElementById("input-field")?.addEventListener("input", (e) => {
  let inputField = e.target as HTMLDivElement
  let content = inputField.innerText
  inputFieldBuffer = content
  if (inputFieldBuffer.trim() === "") {
    spamHeader.classList = ""
    hamHeader.classList = ""
    inputField.innerHTML = ""
    return
  }

  let { outcome: isSpam, positive_probabilities, negative_probabilities } = predict(inputFieldBuffer)
  spamHeader.classList = isSpam ? "lift" : "sink"
  hamHeader.classList = isSpam ? "sink" : "lift"

  updateTokenProbabilitiesUI(inputField, positive_probabilities, negative_probabilities)
})

const preprocessText = (text: string): string => {
  const TRAILING_PUNCTUATION_REGEX = /(?:\r?\n)+|\w+|\p{P}+/gu
  const EMAIL_REGEX = /[a-zA-Z.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*/
  const EMOTICON_REGEX = /[:;=8]-?[)(DPOpo|\\/*]|\([a-z]{2,10}\)/
  const KEEP_REGEX = new RegExp(
    `(${EMAIL_REGEX.source})|(${EMOTICON_REGEX.source})|(${TRAILING_PUNCTUATION_REGEX.source})|([a-zA-Z0-9\\s!?$%'])|(.)`,
    'gu'
  );

  let processed = tokenise(text).map(t => t.toLowerCase()).join(" ")
  const result = processed.replaceAll(
    KEEP_REGEX,
    (match: string, _p1?: string, _p2?: string, _p3?: string, p4?: string): string => {
      return p4 === undefined ? match : " ";
    }
  );
  return result
}

const updateTokenProbabilitiesUI = (inputField: Element, positive_probabilities: Float64Array<ArrayBufferLike>, negative_probabilities: Float64Array<ArrayBufferLike>): void => {
  const caretOffset = getCaretOffset(inputField)
  let t = preprocessText(inputFieldBuffer)

  const tokens = inputFieldBuffer.match(/\w+|\p{P}+|\s+/gu) ?? []
  let processed_tokens = t.split(/\s+/) ?? []
  console.log(positive_probabilities)
  console.log(tokens.length, processed_tokens.length, positive_probabilities.length)

  let output = ""
  let counter = 0
  for (const token of tokens) {
    const newlineCount = (token.match(/\r\n|\n/g) ?? []).length
    if (newlineCount > 0) {
      output += "<br>".repeat(newlineCount)
      continue
    }

    if (/\s+/.test(token)) {
      output += token
      continue
    }

    if (processed_tokens.includes(token)) {
      console.log(positive_probabilities[counter])
      console.log(negative_probabilities[counter])

      output += `<button class="token highlight-yellow" data-tip="Positive: ${parseFloat(positive_probabilities[counter]!.toFixed(6))} Negative: ${parseFloat(negative_probabilities[counter]!.toFixed(6))}"
>${token}</button>`
      counter += 1
      continue
    }
    output += `<button class="token highlight-yellow">${token}</button>`
  }

  inputField.innerHTML = output;
  moveCaretToOffset(inputField, caretOffset)
}

const tokenise = (text: string): string[] => {
  const TRAILING_PUNCTUATION_REGEX = /(?:\r?\n)+|\w+|\p{P}+/gu
  return text.match(TRAILING_PUNCTUATION_REGEX) ?? []
}

const getCaretOffset = (element: Element): number => {
  const selection = window.getSelection()
  if (!selection || !selection.rangeCount) {
    return 0
  }

  const range = selection.getRangeAt(0).cloneRange()
  range.selectNodeContents(element)
  range.setEnd(selection.getRangeAt(0).endContainer, selection.getRangeAt(0).endOffset)
  return range.toString().length
}

const moveCaretToOffset = (element: Element, offset: number): void => {
  const range = document.createRange()
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
  let node, count = 0
  while ((node = walker.nextNode())) {
    const next = count + (node as Text).length

    if (offset <= next) {
      // Found where the caret needs to be moved to 
      range.setStart(node, offset - count)
      range.collapse(true)
      const selection = window.getSelection()
      selection?.removeAllRanges()
      selection?.addRange(range)
      return
    }
    count = next
  }

  // If the offset is past the end, place the caret at the end of the content
  range.selectNodeContents(element)
  range.collapse(false)
  const selection = window.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
}