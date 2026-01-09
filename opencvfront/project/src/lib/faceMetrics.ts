import type { NormalizedLandmark } from "@mediapipe/face_mesh";

export interface FaceMetrics {
  eye_openness: number; // 0..1
  brow_tension: number; // 0..1
  jaw_tension: number; // 0..1
  facial_asymmetry: number; // 0..1
  head_motion: number; // 0..1 (placeholder)
  facial_stress_score: number; // 0..100
}

// Helper: distance between two landmarks
function dist(a: NormalizedLandmark, b: NormalizedLandmark) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Clamp to range
function clamp(v: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, v));
}

// Indices from MediaPipe FaceMesh reference (approximate commonly used points)
const LEFT_EYE_TOP = 386;
const LEFT_EYE_BOTTOM = 374;
const RIGHT_EYE_TOP = 159;
const RIGHT_EYE_BOTTOM = 145;

const UPPER_LIP = 13;
const LOWER_LIP = 14;

const LEFT_NOSE = 98; // left midface
const RIGHT_NOSE = 327; // right midface

const LEFT_BROW = 70; // approx left eyebrow
const RIGHT_BROW = 300; // approx right eyebrow
const LEFT_EYE_CENTER = 468; // iris center if refineLandmarks enabled; fallback to top
const RIGHT_EYE_CENTER = 473;

export function computeFaceMetrics(landmarks: NormalizedLandmark[], prev?: FaceMetrics): FaceMetrics {
  // Normalize distances relative to face width
  const faceWidth = dist(landmarks[LEFT_NOSE], landmarks[RIGHT_NOSE]);
  const norm = (d: number) => (faceWidth > 0 ? d / faceWidth : d);

  // Eye openness: average of vertical distances of both eyes (larger => more open)
  const leftEyeOpen = norm(dist(landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM]));
  const rightEyeOpen = norm(dist(landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM]));
  let eye_openness = clamp((leftEyeOpen + rightEyeOpen) / 2 * 3); // scale for sensitivity

  // Brow tension: distance from brow to eye center (smaller => more tension), invert and clamp
  const leftEyeCenter = landmarks[LEFT_EYE_CENTER] ?? landmarks[LEFT_EYE_TOP];
  const rightEyeCenter = landmarks[RIGHT_EYE_CENTER] ?? landmarks[RIGHT_EYE_TOP];
  const leftBrowGap = norm(dist(landmarks[LEFT_BROW], leftEyeCenter));
  const rightBrowGap = norm(dist(landmarks[RIGHT_BROW], rightEyeCenter));
  let brow_gap = (leftBrowGap + rightBrowGap) / 2;
  let brow_tension = clamp(1 - brow_gap * 3);

  // Jaw tension: mouth opening (smaller => more tension); invert
  const mouthOpen = norm(dist(landmarks[UPPER_LIP], landmarks[LOWER_LIP]));
  let jaw_tension = clamp(1 - mouthOpen * 5);

  // Facial asymmetry: difference between left/right midface symmetry
  const midYLeft = landmarks[LEFT_NOSE].y;
  const midYRight = landmarks[RIGHT_NOSE].y;
  let facial_asymmetry = clamp(Math.abs(midYLeft - midYRight) * 5);

  // Head motion: simple frame-to-frame change (placeholder using prev values)
  let head_motion = clamp(prev ? Math.abs(prev.eye_openness - eye_openness) + Math.abs(prev.brow_tension - brow_tension) : 0);

  // Stress score: weighted combination (tunable)
  const stressRaw = 0.15 * (1 - eye_openness) + 0.35 * brow_tension + 0.25 * jaw_tension + 0.15 * facial_asymmetry + 0.10 * head_motion;
  const facial_stress_score = clamp(stressRaw, 0, 1) * 100;

  return {
    eye_openness,
    brow_tension,
    jaw_tension,
    facial_asymmetry,
    head_motion,
    facial_stress_score,
  };
}
