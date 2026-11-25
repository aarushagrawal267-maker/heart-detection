import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.QRS_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: backend, changeOrigin: true } },
  },
  preview: { port: 4173, proxy: { "/api": { target: backend, changeOrigin: true } } },
  // Plotly (about 4.7 MB) lives in its own lazily loaded chunk; only chart pages fetch it.
  build: { chunkSizeWarningLimit: 5000 },
});
