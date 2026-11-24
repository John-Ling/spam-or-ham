const WORDS_NEWLINES_OR_SYMBOLS = /(?:\r?\n)+|\w+|[\p{General_Category=Punctuation}\p{General_Category=Symbol}]|\s+/gu
const EMAIL_REGEX = /[a-zA-Z.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*/
const EMOTICON_REGEX = /[:;=8]-?[)(DPOpo|\\/*]|\([a-z]{2,10}\)/

type FloatArray = Float64Array<ArrayBufferLike>

export const tokenise = (text: string): string[] => {
  // const TRAILING_PUNCTUATION_REGEX = /(?:\r?\n)+|\w+|\p{P}+/gu
  const TOKEN_REGEX = new RegExp(
    `(${EMAIL_REGEX.source})|(${EMOTICON_REGEX.source})|(${WORDS_NEWLINES_OR_SYMBOLS.source})`,
    'gu'
  );

  const tokens = text.match(TOKEN_REGEX)
  if (!tokens) {
    return []
  }

  return tokens.map(t => t.toLowerCase())
}

export const updateProbabilityCounterUI = (positiveProbability: number, negativeProbability: number) => {
  const positiveCounter = document.getElementById("positive-value")
  const negativeCounter = document.getElementById("negative-value")
  if (!positiveCounter || !negativeCounter) {
    console.error("Cannot update counters")
    return
  }

  positiveCounter.innerText = `${parseFloat((positiveProbability * 100).toFixed(6))}%`
  negativeCounter.innerText = `${parseFloat((negativeProbability * 100).toFixed(6))}%`
}

export const generateOutputHTML = (content: string, positiveLikelihoods: FloatArray, negativeLikelihoods: FloatArray): string => {
  const tokens = tokenise(content)
  let output = ""
  let counter = 0
  for (const token of tokens) {
    if (/\s+/.test(token)) {
      output += token
      continue
    }

    output += `<button data-id="${counter + 1}"  class="token has-tooltip highlight-yellow" data-tip="POSITIVE: ${parseFloat(positiveLikelihoods[counter]!.toFixed(6))} | NEGATIVE: ${parseFloat(negativeLikelihoods[counter]!.toFixed(6))}">${token}</button>`
    counter += 1
  }
  return output
}

export const getCaretOffset = (element: Element): number => {
  const selection = window.getSelection()
  if (!selection || !selection.rangeCount) {
    return 0
  }

  const range = selection.getRangeAt(0).cloneRange()
  range.selectNodeContents(element)
  range.setEnd(selection.getRangeAt(0).endContainer, selection.getRangeAt(0).endOffset)
  return range.toString().length
}

export const moveCaretToOffset = (element: Element, offset: number): void => {
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