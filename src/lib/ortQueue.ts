/**
 * Serializes onnxruntime-web work across every session in the page.
 *
 * The alphabet and CTC models hold separate sessions but share one wasm
 * backend. Overlapping runs throw "Session already started" or "Session
 * mismatch", and overlapping session creation throws "multiple calls to
 * initWasm()" because the backend initializes lazily on the first create.
 * Both paths go through the same queue so only one touches the backend at a
 * time
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
