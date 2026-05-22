import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

// 127-landmark Holistic subset. Layout matches openhand-model's
// signs_landmarks.py. We only render the two hands.
const N_LANDMARKS = 127;

// Group offsets in landmark units (matches signs_landmarks.py).
const LEFT_HAND_START = 76 + 9;       // face + pose offset
const RIGHT_HAND_START = LEFT_HAND_START + 21;
const HAND_LEN = 21;

// MediaPipe hand bone topology, applied to each hand.
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

interface Props {
  /** Flat T*127*3 normalized landmark sequence from /api/sign-references. */
  landmarks: ReadonlyArray<number>;
  /** Flat T*127 missing mask (0 = present, 1 = absent). */
  missing: ReadonlyArray<number>;
  /** Number of frames. */
  frames: number;
  /** Playback rate in frames per second. Defaults to 12. */
  fps?: number;
  color?: string;
}

interface FrameTensor {
  /** (127, 3) per-frame Vector3 points. */
  points: THREE.Vector3[];
  /** (127,) per-frame boolean mask. */
  missing: boolean[];
}

function unpack(
  landmarks: ReadonlyArray<number>,
  missing: ReadonlyArray<number>,
  T: number,
): FrameTensor[] {
  const out: FrameTensor[] = [];
  for (let t = 0; t < T; t++) {
    const points: THREE.Vector3[] = [];
    const miss: boolean[] = [];
    for (let lm = 0; lm < N_LANDMARKS; lm++) {
      const i = (t * N_LANDMARKS + lm) * 3;
      // MediaPipe is x-right, y-down. Flip y so the hand stands upright
      // in the y-up Three.js world.
      const x = landmarks[i];
      const y = -landmarks[i + 1];
      const z = landmarks[i + 2];
      points.push(new THREE.Vector3(x, y, z));
      miss.push(missing[t * N_LANDMARKS + lm] !== 0);
    }
    out.push({ points, missing: miss });
  }
  return out;
}

function Bones({ points, missing, color }: { points: THREE.Vector3[]; missing: boolean[]; color: string }) {
  return (
    <>
      {[LEFT_HAND_START, RIGHT_HAND_START].map((handStart) =>
        HAND_CONNECTIONS.map(([a, b], i) => {
          if (missing[handStart + a] || missing[handStart + b]) return null;
          const pa = points[handStart + a];
          const pb = points[handStart + b];
          const mid = pa.clone().add(pb).multiplyScalar(0.5);
          const dir = pb.clone().sub(pa);
          const len = dir.length();
          if (len < 1e-4 || !Number.isFinite(len)) return null;
          const quat = new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            dir.clone().normalize(),
          );
          return (
            <mesh key={`${handStart}-${i}`} position={mid} quaternion={quat}>
              <cylinderGeometry args={[0.015, 0.015, len, 8]} />
              <meshStandardMaterial color={color} roughness={0.4} metalness={0.1} />
            </mesh>
          );
        }),
      )}
    </>
  );
}

function HandJoints({ points, missing, color }: { points: THREE.Vector3[]; missing: boolean[]; color: string }) {
  const items: { p: THREE.Vector3; key: number }[] = [];
  for (const handStart of [LEFT_HAND_START, RIGHT_HAND_START]) {
    for (let i = 0; i < HAND_LEN; i++) {
      const lm = handStart + i;
      if (missing[lm]) continue;
      items.push({ p: points[lm], key: lm });
    }
  }
  return (
    <>
      {items.map(({ p, key }) => (
        <mesh key={key} position={p}>
          <sphereGeometry args={[0.022, 14, 14]} />
          <meshStandardMaterial color={color} roughness={0.3} metalness={0.2} />
        </mesh>
      ))}
    </>
  );
}

function Scene({
  framesData,
  fps,
  color,
}: {
  framesData: FrameTensor[];
  fps: number;
  color: string;
}) {
  const [frameIdx, setFrameIdx] = useState(0);
  const accumRef = useRef(0);
  const T = framesData.length;

  // Compute a stable anchor across the whole clip so the camera doesn't
  // jitter frame to frame. We center on the average hand position over
  // the whole sequence.
  const anchor = useMemo(() => {
    const a = new THREE.Vector3();
    let n = 0;
    for (const f of framesData) {
      for (let i = 0; i < HAND_LEN; i++) {
        if (!f.missing[LEFT_HAND_START + i]) {
          a.add(f.points[LEFT_HAND_START + i]);
          n++;
        }
        if (!f.missing[RIGHT_HAND_START + i]) {
          a.add(f.points[RIGHT_HAND_START + i]);
          n++;
        }
      }
    }
    if (n > 0) a.multiplyScalar(1 / n);
    return a;
  }, [framesData]);

  useFrame((_, delta) => {
    accumRef.current += delta;
    const frameDur = 1.0 / Math.max(fps, 1);
    while (accumRef.current >= frameDur) {
      accumRef.current -= frameDur;
      setFrameIdx((i) => (T > 0 ? (i + 1) % T : 0));
    }
  });

  // Clamp frameIdx into range so this stays defined when framesData changes
  // (e.g. switching to a shorter clip). All hooks must run unconditionally.
  const safeIdx = T > 0 ? Math.min(frameIdx, T - 1) : 0;
  const current = framesData[safeIdx];

  // Apply the stable anchor offset to every point this frame.
  const shifted = useMemo(
    () => current ? current.points.map((p) => p.clone().sub(anchor)) : [],
    [current, anchor],
  );

  // Frame the camera once.
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(0, 0, 4);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  if (!current) return null;

  return (
    <group>
      <Bones points={shifted} missing={current.missing} color={color} />
      <HandJoints points={shifted} missing={current.missing} color={color} />
    </group>
  );
}

export function SignAnimation3D({
  landmarks,
  missing,
  frames,
  fps = 12,
  color = "#3c82f0",
}: Props) {
  const framesData = useMemo(
    () => unpack(landmarks, missing, frames),
    [landmarks, missing, frames],
  );

  return (
    <Canvas
      camera={{ position: [0, 0, 4], fov: 45 }}
      dpr={[1, 2]}
      style={{ width: "100%", height: "100%" }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[3, 4, 5]} intensity={0.8} />
      <directionalLight position={[-3, -2, -4]} intensity={0.25} />
      <Scene framesData={framesData} fps={fps} color={color} />
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={2}
        maxDistance={8}
        autoRotate={false}
      />
    </Canvas>
  );
}
