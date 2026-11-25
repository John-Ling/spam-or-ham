/* tslint:disable */
/* eslint-disable */

export class PredictionResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    negative_probability: number;
    outcome: boolean;
    positive_probability: number;
    readonly negative_likelihoods: Float64Array;
    readonly positive_likelihoods: Float64Array;
}

export function load_classifier(model_json_string: string): boolean;

export function predict(string: string): PredictionResult;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_get_predictionresult_negative_probability: (a: number) => number;
    readonly __wbg_get_predictionresult_outcome: (a: number) => number;
    readonly __wbg_get_predictionresult_positive_probability: (a: number) => number;
    readonly __wbg_predictionresult_free: (a: number, b: number) => void;
    readonly __wbg_set_predictionresult_negative_probability: (a: number, b: number) => void;
    readonly __wbg_set_predictionresult_outcome: (a: number, b: number) => void;
    readonly __wbg_set_predictionresult_positive_probability: (a: number, b: number) => void;
    readonly load_classifier: (a: number, b: number) => number;
    readonly predict: (a: number, b: number) => number;
    readonly predictionresult_negative_likelihoods: (a: number) => [number, number];
    readonly predictionresult_positive_likelihoods: (a: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
