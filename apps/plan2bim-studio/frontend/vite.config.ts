import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.DAJOONG_EMBED_WEB === "true" ? "/studio/" : "/",
  plugins: [react()],
  css: {
    // Keep Studio builds isolated from user-level PostCSS and Tailwind configs.
    postcss: { plugins: [] },
  },
  build: {
    chunkSizeWarningLimit: 550,
    rollupOptions: {
      output: {
        manualChunks: {
          auth: ["oidc-client-ts"],
          react: ["react", "react-dom"],
          three: ["three", "three/examples/jsm/controls/OrbitControls.js"],
          icons: ["lucide-react"],
        },
      },
    },
  },
  server: { port: 5173 },
});
