/**
 * Shared execution-provider config for onnxruntime-web sessions.
 *
 * Prefers WebGPU and falls back to wasm when the browser has no GPU. Logs each
 * session's preference so the console shows which backend it asked for
 */

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

// Small models run on WebGPU when available, falling back to wasm
export const executionProviders = hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];

// The CTC model is large and has hung the GPU device (an unrecoverable page
// freeze) on the WebGPU path. wasm runs it reliably
export const ctcExecutionProviders = ["wasm"];

const _logged = new Set<string>();

/** Log the chosen provider preference once per session label */
export function logProvider(label: string, providers = executionProviders): void {
  if (_logged.has(label)) return;
  _logged.add(label);
  console.info(
    `[onnx] WebGPU ${hasWebGPU ? "available" : "unavailable"}, ${label} session requesting [${providers.join(", ")}]`,
  );
}
