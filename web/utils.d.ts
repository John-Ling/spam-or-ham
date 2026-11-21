export declare const preprocessText: (text: string) => string;
export declare const updateTokenProbabilitiesUI: ({ content, inputField, positive_probabilities, negative_probabilities }: {
    content: string;
    inputField: Element;
    positive_probabilities: Float64Array<ArrayBufferLike>;
    negative_probabilities: Float64Array<ArrayBufferLike>;
}) => void;
export declare const generateOutputHTML: (content: string, positive_probabilities: Float64Array<ArrayBufferLike>, negative_probabilities: Float64Array<ArrayBufferLike>) => string;
export declare const tokenise: (text: string) => string[];
export declare const moveCaretToOffset: (element: Element, offset: number) => void;
//# sourceMappingURL=utils.d.ts.map