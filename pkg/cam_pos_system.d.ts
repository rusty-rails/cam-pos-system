/* tslint:disable */
/* eslint-disable */
/**
* @param {Element} root
*/
export function main(root: Element): void;
/**
*/
export class MultiMosseTrackerJS {
  free(): void;
/**
* @param {number} width
* @param {number} height
*/
  constructor(width: number, height: number);
/**
* @param {number} x
* @param {number} y
* @param {Uint8Array} img_data
*/
  set_target(x: number, y: number, img_data: Uint8Array): void;
/**
* @param {Uint8Array} img_data
* @returns {Uint8Array}
*/
  track(img_data: Uint8Array): Uint8Array;
}
/**
*/
export class TrackerJS {
  free(): void;
/**
* @param {number} width
* @param {number} height
*/
  constructor(width: number, height: number);
/**
* @param {number} x
* @param {number} y
* @param {Uint8Array} img_data
*/
  set_target(x: number, y: number, img_data: Uint8Array): void;
/**
* @param {Uint8Array} img_data
* @returns {Uint8Array}
*/
  next(img_data: Uint8Array): Uint8Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_trackerjs_free: (a: number) => void;
  readonly trackerjs_new: (a: number, b: number) => number;
  readonly trackerjs_set_target: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly trackerjs_next: (a: number, b: number, c: number, d: number) => void;
  readonly main: (a: number) => void;
  readonly __wbg_multimossetrackerjs_free: (a: number) => void;
  readonly multimossetrackerjs_new: (a: number, b: number) => number;
  readonly multimossetrackerjs_set_target: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly multimossetrackerjs_track: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly _dyn_core__ops__function__Fn__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__h007cd4f9e1e5466d: (a: number, b: number, c: number) => void;
  readonly _dyn_core__ops__function__FnMut__A____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__hea741bfcb340add5: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly wasm_bindgen__convert__closures__invoke3_mut__hdd84371aa6ef3b7d: (a: number, b: number, c: number, d: number, e: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
