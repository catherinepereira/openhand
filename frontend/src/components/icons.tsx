export function HandIcon() {
  return (
    <svg
      width="48"
      height="48"
      viewBox="0 0 24 24"
      fill="none"
      stroke="#b0b0b0"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 11V7a2 2 0 0 0-4 0v4" />
      <path d="M14 10V5a2 2 0 0 0-4 0v5" />
      <path d="M10 10.5V4a2 2 0 0 0-4 0v8" />
      <path d="M6 14a4 4 0 0 0 4 4h4a4 4 0 0 0 4-4v-3H6v3z" />
    </svg>
  );
}

/** Lucide "eraser" outline glyph; inlined so we don't pull in lucide-react. */
export function EraserIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m7 21 3-3-7-7 6-6c.6-.6 1.6-.6 2.2 0l6.8 6.8c.6.6.6 1.6 0 2.2L11 21H7Z" />
      <path d="M22 21H7" />
      <path d="m5 11 9 9" />
    </svg>
  );
}
