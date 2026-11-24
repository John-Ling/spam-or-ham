type FloatArray = Float64Array<ArrayBufferLike>;
export declare const tokenise: (text: string) => string[];
export declare const updateProbabilityCounterUI: (positiveProbability: number, negativeProbability: number) => void;
export declare const generateOutputHTML: (content: string, positiveLikelihoods: FloatArray, negativeLikelihoods: FloatArray) => string;
export declare const getCaretOffset: (element: Element) => number;
export declare const moveCaretToOffset: (element: Element, offset: number) => void;
export {};
//# sourceMappingURL=utils.d.ts.map