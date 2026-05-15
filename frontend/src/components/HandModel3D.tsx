import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

// MediaPipe hand topology: 21 landmarks per hand, indexed 0..20.
// Connections form the 5 fingers + palm. Same set used by WebcamFeed.tsx.
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

interface Props {
  /** Length-63 array of 21 (x, y, z) MediaPipe landmarks, wrist-anchored
   *  and p95-scaled (same coordinate space the alphabet classifier sees). */
  landmarks: ReadonlyArray<number>;
  /** Optional accent color for joints and bones. */
  color?: string;
  /** If true, the model auto-rotates around the Y axis. Useful for the
   *  reference panel to give a sense of depth. */
  autoRotate?: boolean;
}

/**
 * Convert the flat 63-float MediaPipe vector to 21 THREE.Vector3 points.
 * MediaPipe is x-right, y-down, z-toward-camera; we flip y so the hand
 * stands upright in Three.js's default y-up world.
 */
function toPoints(landmarks: ReadonlyArray<number>): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < 21; i++) {
    const x = landmarks[i * 3];
    const y = -landmarks[i * 3 + 1];
    const z = landmarks[i * 3 + 2];
    pts.push(new THREE.Vector3(x, y, z));
  }
  return pts;
}

function Bones({ points, color }: { points: THREE.Vector3[]; color: string }) {
  // Each bone is a thin cylinder oriented from joint a to joint b.
  return (
    <>
      {HAND_CONNECTIONS.map(([a, b], i) => {
        const pa = points[a];
        const pb = points[b];
        const mid = pa.clone().add(pb).multiplyScalar(0.5);
        const dir = pb.clone().sub(pa);
        const len = dir.length();
        const quat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          dir.clone().normalize(),
        );
        return (
          <mesh
            key={i}
            position={mid}
            quaternion={quat}
          >
            <cylinderGeometry args={[0.02, 0.02, len, 8]} />
            <meshStandardMaterial color={color} roughness={0.4} metalness={0.1} />
          </mesh>
        );
      })}
    </>
  );
}

function Joints({ points, color }: { points: THREE.Vector3[]; color: string }) {
  return (
    <>
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[i === 0 ? 0.05 : 0.035, 16, 16]} />
          <meshStandardMaterial color={color} roughness={0.3} metalness={0.2} />
        </mesh>
      ))}
    </>
  );
}

function Scene({ points, color, autoRotate }: { points: THREE.Vector3[]; color: string; autoRotate: boolean }) {
  const groupRef = useRef<THREE.Group>(null);

  // Center the hand around the wrist (already at origin given the
  // wrist-anchored normalization, but be defensive in case input drifts).
  const centered = useMemo(() => {
    const wrist = points[0];
    return points.map((p) => p.clone().sub(wrist));
  }, [points]);

  useFrame((_, delta) => {
    if (autoRotate && groupRef.current) {
      groupRef.current.rotation.y += delta * 0.4;
    }
  });

  // On first render, frame the camera so the hand fills the viewport.
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(0, 0, 4);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return (
    <group ref={groupRef}>
      <Bones points={centered} color={color} />
      <Joints points={centered} color={color} />
    </group>
  );
}

export function HandModel3D({ landmarks, color = "#3c82f0", autoRotate = true }: Props) {
  const points = useMemo(() => toPoints(landmarks), [landmarks]);

  return (
    <Canvas
      camera={{ position: [0, 0, 4], fov: 45 }}
      dpr={[1, 2]}
      style={{ width: "100%", height: "100%" }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[3, 4, 5]} intensity={0.8} />
      <directionalLight position={[-3, -2, -4]} intensity={0.25} />
      <Scene points={points} color={color} autoRotate={autoRotate} />
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
