/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5273, strictPort: true },
  test: {
    // happy-dom is lighter and ships clean ESM (jsdom 27 + Node 22 hit a
    // require-of-ESM error in jsdom's css-color dependency).
    environment: "happy-dom",
    globals: true,
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
