import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    proxy: { "/api": process.env.API_PROXY_TARGET || "http://127.0.0.1:8501" },
  },
  preview: { host: "0.0.0.0" },
  build: { chunkSizeWarningLimit: 1200 },
});
