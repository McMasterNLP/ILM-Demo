import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // The service mounts everything under /api/v1 (settings.api_prefix),
        // so the client's /api becomes /api/v1 rather than being stripped.
        rewrite: (path) => path.replace(/^\/api/, "/api/v1"),
      },
    },
  },
});
