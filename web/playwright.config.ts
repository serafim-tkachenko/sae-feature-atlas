import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  globalSetup: "./tests/setup.ts",
  fullyParallel: true,
  use: {
    browserName: "chromium",
    viewport: { width: 1280, height: 900 },
    offline: true,
  },
});
