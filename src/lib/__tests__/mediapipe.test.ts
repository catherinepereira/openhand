/**
 * Tests for the pure helpers in lib/mediapipe.ts. Most of the module wraps
 * MediaPipe's async API and isn't easily unit-testable without mocking the
 * SDK; splitHands is the one pure piece worth pinning, since it controls
 * which hand lands in which slot of RawFrameLandmarks.
 */

import { describe, expect, it } from "vitest";

import { splitHands } from "../mediapipe";
import type { NormalizedLandmark } from "../landmarks";

function makeHand(x: number): NormalizedLandmark[] {
  return Array.from({ length: 21 }, (_, i) => ({
    x: x + i * 0.001,
    y: 0.5,
    z: 0,
  }));
}

/** Build a minimal HandLandmarkerResult-shaped object */
function makeResult(
  hands: { label: "Left" | "Right" | "Unknown"; lms: NormalizedLandmark[] }[],
) {
  return {
    landmarks: hands.map((h) => h.lms),
    worldLandmarks: hands.map((h) => h.lms),
    handedness: hands.map((h) => [
      { categoryName: h.label, score: 0.99, index: 0, displayName: h.label },
    ]),
  } as unknown as Parameters<typeof splitHands>[0];
}

describe("splitHands", () => {
  it("returns all-nulls for a null result", () => {
    const out = splitHands(null);
    expect(out.leftHand).toBeNull();
    expect(out.rightHand).toBeNull();
  });

  it("returns all-nulls for an empty result", () => {
    const out = splitHands(makeResult([]));
    expect(out.leftHand).toBeNull();
    expect(out.rightHand).toBeNull();
  });

  it("places a left-labeled hand into the leftHand slot", () => {
    const leftLms = makeHand(0.1);
    const out = splitHands(makeResult([{ label: "Left", lms: leftLms }]));
    expect(out.leftHand).toBe(leftLms);
    expect(out.rightHand).toBeNull();
  });

  it("places a right-labeled hand into the rightHand slot", () => {
    const rightLms = makeHand(0.9);
    const out = splitHands(makeResult([{ label: "Right", lms: rightLms }]));
    expect(out.rightHand).toBe(rightLms);
    expect(out.leftHand).toBeNull();
  });

  it("keeps left and right separate when both are present", () => {
    const leftLms = makeHand(0.1);
    const rightLms = makeHand(0.9);
    const out = splitHands(
      makeResult([
        { label: "Left", lms: leftLms },
        { label: "Right", lms: rightLms },
      ]),
    );
    expect(out.leftHand).toBe(leftLms);
    expect(out.rightHand).toBe(rightLms);
    expect(out.leftHand).not.toBe(out.rightHand);
  });

  it("ignores hands whose category isn't Left or Right", () => {
    const lms = makeHand(0.5);
    const out = splitHands(makeResult([{ label: "Unknown", lms }]));
    expect(out.leftHand).toBeNull();
    expect(out.rightHand).toBeNull();
  });
});
