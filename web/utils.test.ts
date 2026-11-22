import { expect, describe, test, it, beforeEach, vi } from "vitest"
import { generateOutputHTML, tokenise } from "./utils.js"

describe("tokenise", () => {
  it("captures repeated unix newlines", () => {
    expect(tokenise("\n\n")).toEqual(["\n\n"])
  })
  it("captures repeated windows newlines", () => {
    expect(tokenise("\r\n\r\n")).toEqual(["\r\n\r\n"])
  })
  it("captures whitespace", () => {
    expect(tokenise(" ")).toEqual([" "])
  })

  it("captures words", () => {
    expect(tokenise("a b")).toEqual(["a", " ", "b"])
    expect(tokenise("aa bb")).toEqual(["aa", " ", "bb"])

  })
  it("captures all english symbols", () => {
    const symbols = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "{", "}", "|", "\\", ",", ".", "<", ">", "/", "?", "~", "`", "-", "_", "=", "+"]
    expect(tokenise(symbols.join(" "))).toEqual(symbols.flatMap((s, i) => (i === 0 ? [s] : [" ", s])))
  })
  it("captures groups of symbols as individuals", () => {
    expect(tokenise("@@")).toEqual(["@", "@"])
  })
  it("separates symbols from words as", () => {
    expect(tokenise("^a!")).toEqual(["^", "a", "!"])
  })
  it("separates unix newlines from words", () => {
    expect(tokenise("\na\n\n")).toEqual(["\n", "a", "\n\n"])
  })
  it("separates windows newlines from words", () => {
    expect(tokenise("\r\na\r\n\r\n")).toEqual(["\r\n", "a", "\r\n\r\n"])
  })

  it("returns an empty array if no matches exist", () => {
    expect(tokenise("")).toEqual([])
  })

  it("captures emails", () => {
    expect(tokenise("hello@johnling.me")).toEqual(["hello@johnling.me"])
    expect(tokenise("hello@mail.johnling.me")).toEqual(["hello@mail.johnling.me"])
    expect(tokenise("hello.there@mail.johnling.me")).toEqual(["hello.there@mail.johnling.me"])
    expect(tokenise("hello hello@johnling.me")).toEqual(["hello", " ", "hello@johnling.me"])
    expect(tokenise("hello hello@mail.johnling.me")).toEqual(["hello", " ", "hello@mail.johnling.me"])
  })
  it("captures emoticons", () => {
    expect(tokenise(":)")).toEqual([":)"])
    expect(tokenise("8)")).toEqual(["8)"])
    expect(tokenise("a :)")).toEqual(["a", " ", ":)"])
  })
})

describe("generateOutputHTML", () => {
  const positive_probabilities: Float64Array<ArrayBufferLike> = Float64Array.of(0.1, 0.2, 0.3)
  const negative_probabilities: Float64Array<ArrayBufferLike> = Float64Array.of(0.3, 0.2, 0.1)
  let parser = new DOMParser()

  const generate = (content: string): string => {
    return generateOutputHTML(content, positive_probabilities, negative_probabilities)
  }

  const toHTML = (content: string): Document => {
    return parser.parseFromString(content, "text/html")
  }

  beforeEach(() => {
    parser = new DOMParser()
  })

  it("does not alter any unix or windows newlines and whitespace present", () => {
    expect(generate("\n\n")).toBe("\n\n")
    expect(generate("\r\n")).toBe("\r\n")
    expect(generate("\n \n")).toBe("\n \n")
    expect(generate("\r\n \r\n")).toBe("\r\n \r\n")
    expect(generate("   ")).toBe("   ")
  })
  it("outputs two spans when provided two words separated by space", () => {
    const output = generate("a b")
    const html = toHTML(output)
    expect(output).not.toContain("\n")

    expect(html.querySelectorAll('[data-id]').length).toBe(2)

    expect(html.querySelector('[data-id="1"]')).not.toBeNull()
    expect(html.querySelector('[data-id="1"]')?.textContent).toBe("a")

    expect(html.querySelector('[data-id="2"]')).not.toBeNull()
    expect(html.querySelector('[data-id="2"]')?.textContent).toBe("b")
  })
  it("outputs two spans and a newline when provided two words separated by newline", () => {
    const output = generate("a\nb")
    const html = toHTML(output)
    expect(output).toContain("\n")

    expect(html.querySelectorAll('[data-id]').length).toBe(2)
    expect(html.querySelector('[data-id="1"]')).not.toBeNull()
    expect(html.querySelector('[data-id="1"]')?.textContent).toBe("a")

    expect(html.querySelector('[data-id="2"]')).not.toBeNull()
    expect(html.querySelector('[data-id="2"]')?.textContent).toBe("b")
  })
  it("separates symbols from words as their own spans", () => {
    let html = toHTML(generate("a!"))

    expect(html.querySelectorAll('[data-id]').length).toBe(2)
    expect(html.querySelector('[data-id="1"]')).not.toBeNull()
    expect(html.querySelector('[data-id="1"]')?.textContent).toBe("a")
    expect(html.querySelector('[data-id="2"]')).not.toBeNull()
    expect(html.querySelector('[data-id="2"]')?.textContent).toBe("!")

    html = toHTML(generate("a!!"))
    expect(html.querySelectorAll('[data-id]').length).toBe(3)
    expect(html.querySelector('[data-id="1"]')).not.toBeNull()
    expect(html.querySelector('[data-id="1"]')?.textContent).toBe("a")
    expect(html.querySelector('[data-id="2"]')).not.toBeNull()
    expect(html.querySelector('[data-id="2"]')?.textContent).toBe("!")
    expect(html.querySelector('[data-id="3"]')).not.toBeNull()
    expect(html.querySelector('[data-id="3"]')?.textContent).toBe("!")


    html = toHTML(generate("!a!"))
    expect(html.querySelectorAll('[data-id]').length).toBe(3)
    expect(html.querySelector('[data-id="1"]')).not.toBeNull()
    expect(html.querySelector('[data-id="1"]')?.textContent).toBe("!")
    expect(html.querySelector('[data-id="2"]')).not.toBeNull()
    expect(html.querySelector('[data-id="2"]')?.textContent).toBe("a")
    expect(html.querySelector('[data-id="3"]')).not.toBeNull()
    expect(html.querySelector('[data-id="3"]')?.textContent).toBe("!")
  })
})