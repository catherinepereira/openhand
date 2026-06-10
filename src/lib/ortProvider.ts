/**
 * Shared execution-provider config for onnxruntime-web sessions.
 *
 * Prefers WebGPU and falls back to wasm when the browser has no GPU. Logs the
 * preference once so it's clear in the console whether WebGPU was available.
 */

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

export const executionProviders = hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];

let _logged = false;

/** Log the chosen provider preference once, on first session init. */
export function logProvider(label: string): void {
  if (_logged) return;
  _logged = true;
  console.info(
    `[onnx] WebGPU ${hasWebGPU ? "available" : "unavailable"}, ${label} session requesting [${executionProviders.join(", ")}]`,
  );
}
