// Centralised backend endpoint config. Override via VITE_API_BASE in
// frontend/.env if the backend runs on a different host/port.

const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8273";

// HTTP and WebSocket variants of the same origin.
const WS_BASE: string = API_BASE.replace(/^http/, "ws");

export const HTTP_ENDPOINTS = {
  tts: `${API_BASE}/api/tts`,
  health: `${API_BASE}/api/health`,
};

export const WS_ENDPOINTS = {
  detect: `${WS_BASE}/ws/detect-landmarks`,
  transcribeStream: `${WS_BASE}/ws/transcribe-stream`,
};
