import { expect, describe, test, it } from "vitest";
import { tokenise } from "./utils.js";
describe("tokenise", () => {
    it("captures repeated unix newlines", () => {
        expect(tokenise("\n\n")).toEqual(["\n\n"]);
    });
    it("captures words", () => {
        expect(tokenise("hello world")).toEqual(["hello", "world"]);
    });
    it("captures all english symbols", () => {
        expect(tokenise("! @ # $ % ^ & * ( ) { } | \\ , . < > / ? ~ ` - _ = +")).toEqual(["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "{", "}", "|", "\\", ",", ".", "<", ">", "/", "?", "~", "`", "-", "_", "=", "+"]);
    });
    it("captures groups of symbols as individuals", () => {
        expect(tokenise("@@")).toEqual(["@", "@"]);
    });
    it("separates symbols from words as", () => {
        expect(tokenise("^a!")).toEqual(["^", "a", "!"]);
    });
    it("separates whitespace from words", () => {
        expect(tokenise("\na\n\n")).toEqual(["\n", "a", "\n\n"]);
    });
    it("returns an empty array if no matches exist", () => {
        expect(tokenise("")).toEqual([]);
    });
});
//# sourceMappingURL=script.test.js.map