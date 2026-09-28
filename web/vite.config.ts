import { defineConfig } from "vite";

export default defineConfig({
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  build: {
    outDir: "../src/sae_feature_atlas/report/static",
    emptyOutDir: true,
    cssCodeSplit: false,
    lib: {
      entry: "src/main.tsx",
      name: "AtlasExplorer",
      formats: ["iife"],
      fileName: () => "explorer.js",
      cssFileName: "explorer",
    },
  },
});
