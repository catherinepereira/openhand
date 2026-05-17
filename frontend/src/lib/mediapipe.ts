/**
 * Browser-side MediaPipe Tasks API wrapper.
 *
 * Initializes three detectors (Hand, Pose, Face) once, then exposes per-frame
 * extraction functions that produce typed landmark bundles matching the shape
 * landmarks.ts expects.
 *
 * The three .task files are served by Vite from /models/ (copied at setup
 * time from openhand/backend/models/artifacts/).
 */

import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  FaceLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

import type { NormalizedLandmark, RawFrameLandmarks } from "./landmarks";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

const TASK_URLS = {
  hand: "/models/hand_landmarker.task",
  pose: "/models/pose_landmarker.task",
  face: "/models/face_landmarker.task",
} as const;

// Minimum detection / presence confidences. Match the backend so behavior
// is consistent across the local-Python and in-browser code paths.
const MIN_CONFIDENCE = 0.3;

/** Bundle of initialized detectors; lifetime owned by the caller. */
export interface MediaPipeDetectors {
  hand: HandLandmarker;
  pose: PoseLandmarker;
  face: FaceLandmarker;
  close(): void;
}

/**
 * Create all three detectors in parallel. Resolves once the .task files
 * have been downloaded and the WebAssembly fileset is ready.
 *
 * Pass `running: "video"` (the default) for live-frame use; "image" for
 * one-shot stills.
 */
export async function createDetectors(
  running: "video" | "image" = "image",
): Promise<MediaPipeDetectors> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const runningMode = running.toUpperCase() as "VIDEO" | "IMAGE";

  const [hand, pose, face] = await Promise.all([
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: TASK_URLS.hand },
      numHands: 2,
      minHandDetectionConfidence: MIN_CONFIDENCE,
      minHandPresenceConfidence: MIN_CONFIDENCE,
      minTrackingConfidence: MIN_CONFIDENCE,
      runningMode,
    }),
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: TASK_URLS.pose },
      minPoseDetectionConfidence: MIN_CONFIDENCE,
      minPosePresenceConfidence: MIN_CONFIDENCE,
      minTrackingConfidence: MIN_CONFIDENCE,
      runningMode,
    }),
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: TASK_URLS.face },
      numFaces: 1,
      minFaceDetectionConfidence: MIN_CONFIDENCE,
      minFacePresenceConfidence: MIN_CONFIDENCE,
      minTrackingConfidence: MIN_CONFIDENCE,
      runningMode,
    }),
  ]);

  return {
    hand,
    pose,
    face,
    close() {
      hand.close();
      pose.close();
      face.close();
    },
  };
}

/** One frame's typed detection output, ready to feed into
 *  {@link buildFrameFeatures} from `landmarks.ts`. */
export interface FrameDetection {
  landmarks: RawFrameLandmarks;
}

// Reusable offscreen canvas for sampling the video at its native source
// resolution. Passing the <video> directly to MediaPipe samples it at the
// element's *display* size (CSS-scaled), which shifts coordinates when
// that differs from the underlying stream's videoWidth/Height. Going
// through a canvas at videoWidth x videoHeight matches the backend's
// JPEG-decode path pixel-for-pixel.
const _sampleCanvas: HTMLCanvasElement =
  typeof document !== "undefined" ? document.createElement("canvas") : (null as never);
let _sampleCtx: CanvasRenderingContext2D | null = null;

function sampleVideoAtNativeRes(video: HTMLVideoElement): HTMLCanvasElement | null {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w === 0 || h === 0) return null;
  if (_sampleCanvas.width !== w) _sampleCanvas.width = w;
  if (_sampleCanvas.height !== h) _sampleCanvas.height = h;
  if (_sampleCtx === null) _sampleCtx = _sampleCanvas.getContext("2d");
  if (_sampleCtx === null) return null;
  _sampleCtx.drawImage(video, 0, 0, w, h);
  return _sampleCanvas;
}

/**
 * Run all three detectors on the same video frame and package the output
 * into the shape landmarks.ts expects.
 *
 * Uses IMAGE-mode detect() so each frame is treated independently; matches
 * the backend pipeline (stateless IMAGE detector on a decoded JPEG) and
 * avoids video-mode coordinate drift.
 *
 * timestampMs is unused in IMAGE mode but kept in the signature so callers
 * don't have to change if we swap modes later.
 */
export function detectFrame(
  detectors: MediaPipeDetectors,
  video: HTMLVideoElement,
  _timestampMs: number,
): FrameDetection {
  const source = sampleVideoAtNativeRes(video) ?? video;
  const handResult = safeDetect(() => detectors.hand.detect(source));
  const poseResult = safeDetect(() => detectors.pose.detect(source));
  const faceResult = safeDetect(() => detectors.face.detect(source));

  const { leftHand, rightHand } = splitHands(handResult);
  const pose = firstOrNull(poseResult?.landmarks);
  const face = firstOrNull(faceResult?.faceLandmarks);

  return { landmarks: { face, pose, leftHand, rightHand } };
}

/**
 * Run a MediaPipe detect call and swallow any thrown errors. A failing
 * single sub-detector shouldn't crash the whole frame.
 */
function safeDetect<T>(fn: () => T): T | null {
  try {
    return fn();
  } catch {
    return null;
  }
}

function firstOrNull<T>(arr: readonly T[] | undefined): T | null {
  return arr && arr.length > 0 ? arr[0] : null;
}

/**
 * Split the (up to 2) detected hands into a left + right pair using
 * MediaPipe's handedness classifier.
 *
 * Note this is camera-POV "left/right": a right-handed user with a
 * non-mirrored feed has their right hand appear on the camera's left side,
 * so MediaPipe labels it "Left."
 */
export function splitHands(result: HandLandmarkerResult | null): {
  leftHand: NormalizedLandmark[] | null;
  rightHand: NormalizedLandmark[] | null;
} {
  let leftHand: NormalizedLandmark[] | null = null;
  let rightHand: NormalizedLandmark[] | null = null;

  if (!result || !result.landmarks) {
    return { leftHand, rightHand };
  }

  for (let i = 0; i < result.landmarks.length; i++) {
    const lms = result.landmarks[i];
    const top = result.handedness?.[i]?.[0]?.categoryName;
    if (top === "Left") leftHand = lms;
    else if (top === "Right") rightHand = lms;
  }

  return { leftHand, rightHand };
}
