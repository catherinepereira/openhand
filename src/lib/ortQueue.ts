/**
 * Serializes onnxruntime-web inference across every session in the page.
 *
 * The alphabet and CTC models hold separate sessions but share the wasm
 * backend, and onnxruntime-web throws "Session already started" or "Session
 * mismatch" when two runs overlap on it. Each caller awaits the previous run
 * before starting its own
 */

let tail: Promise<unknown> = Promise.resolve();

/** Run `job` once every previously queued job has settled */
export function runExclusive<T>(job: () => Promise<T>): Promise<T> {
  const result = tail.then(job, job);
  // Swallow rejections on the chain itself so one failed run doesn't reject
  // every job queued behind it
  tail = result.catch(() => undefined);
  return result;
}
