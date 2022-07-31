/* tslint:disable */
/* eslint-disable */
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
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
}

/**
* Synchronously compiles the given `bytes` and instantiates the WebAssembly module.
*
* @param {BufferSource} bytes
*
* @returns {InitOutput}
*/
export function initSync(bytes: BufferSource): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
