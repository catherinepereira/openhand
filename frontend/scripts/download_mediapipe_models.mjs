/**
 * Download MediaPipe Tasks .task model files into frontend/public/models/.
 *
 * Run once after `npm install`. Idempotent — files already on disk are
 * reported but not re-downloaded. The frontend serves these files
 * statically (Vite serves anything in `public/` from the site root) and
 * the MediaPipe Tasks JS loader fetches them at runtime.
 *
 * Usage:
 *   node frontend/scripts/download_mediapipe_models.mjs
 */

import { existsSync, mkdirSync, statSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const PUBLIC_MODELS = resolve(here, "..", "public", "models");
mkdirSync(PUBLIC_MODELS, { recursive: true });

const MODELS = {
  "hand_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  "pose_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  "face_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};

for (const [name, url] of Object.entries(MODELS)) {
  const dest = resolve(PUBLIC_MODELS, name);
  if (existsSync(dest)) {
    const mb = (statSync(dest).size / 1e6).toFixed(1);
    console.log(`OK    ${name} (${mb} MB)`);
    continue;
  }
  process.stdout.write(`DL    ${name} ...`);
  const res = await fetch(url);
  if (!res.ok) {
    console.error(` FAILED (HTTP ${res.status})`);
    process.exit(1);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(dest, buf);
  const mb = (buf.length / 1e6).toFixed(1);
  console.log(` saved ${mb} MB`);
}
