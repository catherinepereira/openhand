/**
 * 127-landmark selection + per-sequence normalization for the CTC
 * fingerspelling model.
 *
 * MUST stay in sync with openhand-model/model/landmarks.py and
 * openhand/backend/services/ctc_landmarks.py. The CTC ONNX has no idea what
 * any of these slots mean; if the order here drifts from training time,
 * the model silently produces nonsense.
 *
 * Type-level design:
 *   - {@link LandmarkGroup} enumerates the four input groups.
 *   - {@link GROUP_INDICES} is the single source of truth for which raw
 *     MediaPipe indices each slot uses, in canonical order.
 *   - {@link FRAME_FEATURES} (= 381) and {@link FRAME_LANDMARKS} (= 127)
 *     are derived; bumping the input lists updates both.
 *   - {@link buildFrameFeatures} consumes a typed {@link RawFrameLandmarks}
 *     bag with one property per group, so swapping inputs becomes a compile
 *     error instead of a runtime slot mix-up.
 */


/** 40 lip outline indices on the MediaPipe FaceMesh. */
export const LIPS_IDX = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
] as const;

/** 16 left-eye outline indices on the MediaPipe FaceMesh. */
export const LEFT_EYE_IDX = [
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
] as const;

/** 16 right-eye outline indices on the MediaPipe FaceMesh. */
export const RIGHT_EYE_IDX = [
  362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
] as const;

/** 4 nose bridge indices on the MediaPipe FaceMesh. */
export const NOSE_IDX = [1, 2, 98, 327] as const;

/**
 * Upper-body pose indices on the MediaPipe PoseLandmarker.
 * 0=nose, 11/12=shoulders, 13/14=elbows, 15/16=wrists, 23/24=hips.
 */
export const POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24] as const;

/** All 21 hand landmark indices (used for both left and right). */
export const HAND_IDX = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
] as const;

/** The four input sources for the CTC model. */
export type LandmarkGroup = "face" | "pose" | "leftHand" | "rightHand";

/**
 * Per-group internal sub-region ordering. Face is composed of four
 * sub-regions concatenated; pose and the two hands are flat. This order
 * must match the training-time concatenation in landmarks.py exactly.
 */
const FACE_SUBGROUPS = [LIPS_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX] as const;
const FACE_INDICES: readonly number[] = FACE_SUBGROUPS.flatMap((g) => [...g]);

/**
 * The raw MediaPipe indices each group draws from, in canonical order.
 * Order of GROUP_INDICES keys matches the order of slots in the output
 * feature vector.
 */
export const GROUP_INDICES = {
  face: FACE_INDICES,
  pose: [...POSE_IDX] as readonly number[],
  leftHand: [...HAND_IDX] as readonly number[],
  rightHand: [...HAND_IDX] as readonly number[],
} as const satisfies Record<LandmarkGroup, readonly number[]>;

/** Number of landmarks per group, in slot order. */
export const GROUP_SIZES: Record<LandmarkGroup, number> = {
  face: GROUP_INDICES.face.length,
  pose: GROUP_INDICES.pose.length,
  leftHand: GROUP_INDICES.leftHand.length,
  rightHand: GROUP_INDICES.rightHand.length,
};

/** Offsets (in *landmark units*, not features) where each group starts. */
export const GROUP_OFFSETS: Record<LandmarkGroup, number> = (() => {
  let cursor = 0;
  const out = {} as Record<LandmarkGroup, number>;
  for (const g of ["face", "pose", "leftHand", "rightHand"] as const) {
    out[g] = cursor;
    cursor += GROUP_SIZES[g];
  }
  return out;
})();

/** Total landmark count across all groups: 40+16+16+4 + 9 + 21 + 21 = 127. */
export const FRAME_LANDMARKS: number =
  GROUP_SIZES.face + GROUP_SIZES.pose + GROUP_SIZES.leftHand + GROUP_SIZES.rightHand;

/** Total floats per frame: 3 (xyz) * FRAME_LANDMARKS = 381. */
export const FRAME_FEATURES: number = FRAME_LANDMARKS * 3;

/** A single MediaPipe landmark in normalized (0..1) coordinates. */
export interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
}

/**
 * The raw landmarks from one webcam frame, organised by source group.
 * Each property is either the full landmark list for that source, or
 * `null` when the corresponding MediaPipe detector found nothing.
 *
 * - {@link face}: the full 468-element FaceMesh array (we index into it via {@link FACE_INDICES})
 * - {@link pose}: the full 33-element pose array
 * - {@link leftHand}, {@link rightHand}: each a 21-element hand array
 *
 * The hand split into left/right is the caller's responsibility; see
 * lib/mediapipe.ts for how we use MediaPipe's handedness output to
 * assign them.
 */
export interface RawFrameLandmarks {
  face: readonly NormalizedLandmark[] | null;
  pose: readonly NormalizedLandmark[] | null;
  leftHand: readonly NormalizedLandmark[] | null;
  rightHand: readonly NormalizedLandmark[] | null;
}

/**
 * One frame's packed features and which landmarks were absent.
 * Both arrays have a fixed length determined by FRAME_FEATURES /
 * FRAME_LANDMARKS at compile time.
 */
export interface FrameFeatures {
  /** length FRAME_FEATURES (= 381) */
  readonly features: Float32Array;
  /** length FRAME_LANDMARKS (= 127). 1 where landmark was absent. */
  readonly missing: Uint8Array;
}

/**
 * Pack one frame's landmarks into the FRAME_FEATURES-float feature vector.
 *
 * The output ordering is fixed by GROUP_INDICES and matches the training-
 * time canonical column ordering. Missing groups produce all-zero feature
 * slots plus `missing[i] = 1` for each landmark.
 */
export function buildFrameFeatures(input: RawFrameLandmarks): FrameFeatures {
  const features = new Float32Array(FRAME_FEATURES);
  const missing = new Uint8Array(FRAME_LANDMARKS);

  let lm = 0;
  for (const group of ["face", "pose", "leftHand", "rightHand"] as const) {
    const source = input[group];
    const indices = GROUP_INDICES[group];
    if (source === null) {
      for (let i = 0; i < indices.length; i++) {
        missing[lm + i] = 1;
      }
    } else {
      for (let i = 0; i < indices.length; i++) {
        const sourceIdx = indices[i];
        const point = source[sourceIdx];
        const base = (lm + i) * 3;
        features[base] = point.x;
        features[base + 1] = point.y;
        features[base + 2] = point.z;
      }
    }
    lm += indices.length;
  }
  return { features, missing };
}

// Sanity check that the four group offsets cover the full landmark range
// with no gaps or overlaps. Catches drift between GROUP_INDICES order and
// GROUP_OFFSETS construction order.
if (GROUP_OFFSETS.rightHand + GROUP_SIZES.rightHand !== FRAME_LANDMARKS) {
  throw new Error(
    `Landmark group offsets out of sync: rightHand ends at ${
      GROUP_OFFSETS.rightHand + GROUP_SIZES.rightHand
    }, expected ${FRAME_LANDMARKS}`,
  );
}

/**
 * Normalize a sequence of (T, FRAME_FEATURES) frames in place, matching the
 * training-time normalize_sequence formula:
 *
 *   1. Pick the dominant-hand wrist (whichever is present in more frames)
 *      and anchor all landmarks to its mean position across the sequence.
 *   2. Divide by the 95th-percentile absolute deviation of all present
 *      landmark coordinates.
 *   3. Re-zero landmarks that were originally missing.
 *
 * @param features (T, FRAME_FEATURES) tensor, flattened row-major.
 *                  Modified in place.
 * @param missing  (T, FRAME_LANDMARKS) byte mask. Read-only.
 * @param T        number of frames.
 */
export function normalizeSequence(
  features: Float32Array,
  missing: Uint8Array,
  T: number,
): Float32Array {
  if (T === 0) return features;

  const rWristLm = GROUP_OFFSETS.rightHand;
  const lWristLm = GROUP_OFFSETS.leftHand;
  let rPresent = 0;
  let lPresent = 0;
  for (let t = 0; t < T; t++) {
    if (missing[t * FRAME_LANDMARKS + rWristLm] === 0) rPresent++;
    if (missing[t * FRAME_LANDMARKS + lWristLm] === 0) lPresent++;
  }
  const useRight = rPresent >= lPresent;
  const wristLm = useRight ? rWristLm : lWristLm;

  let ax = 0;
  let ay = 0;
  let az = 0;
  let nPresent = 0;
  for (let t = 0; t < T; t++) {
    if (missing[t * FRAME_LANDMARKS + wristLm] !== 0) continue;
    const base = (t * FRAME_LANDMARKS + wristLm) * 3;
    ax += features[base];
    ay += features[base + 1];
    az += features[base + 2];
    nPresent++;
  }
  if (nPresent > 0) {
    ax /= nPresent;
    ay /= nPresent;
    az /= nPresent;
  }

  for (let t = 0; t < T; t++) {
    for (let lmIdx = 0; lmIdx < FRAME_LANDMARKS; lmIdx++) {
      const base = (t * FRAME_LANDMARKS + lmIdx) * 3;
      features[base] -= ax;
      features[base + 1] -= ay;
      features[base + 2] -= az;
    }
  }

  // 95th-percentile absolute value across present landmark coordinates.
  const presentVals: number[] = [];
  for (let t = 0; t < T; t++) {
    for (let lmIdx = 0; lmIdx < FRAME_LANDMARKS; lmIdx++) {
      if (missing[t * FRAME_LANDMARKS + lmIdx] !== 0) continue;
      const base = (t * FRAME_LANDMARKS + lmIdx) * 3;
      presentVals.push(
        Math.abs(features[base]),
        Math.abs(features[base + 1]),
        Math.abs(features[base + 2]),
      );
    }
  }
  let scale = 1;
  if (presentVals.length > 0) {
    presentVals.sort((a, b) => a - b);
    // Match numpy.percentile(arr, 95) default "linear" interpolation.
    const pos = 0.95 * (presentVals.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    const frac = pos - lo;
    scale = presentVals[lo] + frac * (presentVals[hi] - presentVals[lo]);
  }
  if (scale > 1e-6) {
    const inv = 1 / scale;
    for (let i = 0; i < features.length; i++) features[i] *= inv;
  }

  // Re-zero missing landmarks; anchor subtraction made them non-zero.
  for (let t = 0; t < T; t++) {
    for (let lmIdx = 0; lmIdx < FRAME_LANDMARKS; lmIdx++) {
      if (missing[t * FRAME_LANDMARKS + lmIdx] === 0) continue;
      const base = (t * FRAME_LANDMARKS + lmIdx) * 3;
      features[base] = 0;
      features[base + 1] = 0;
      features[base + 2] = 0;
    }
  }

  return features;
}
