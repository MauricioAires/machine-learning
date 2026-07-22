import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 5000,
  expect: {
    timeout: 5000,
  },
  use: {
    baseURL: "https://erickwendel.github.io",
    trace: "on-first-retry",
  },
  reporter: [["list"], ["html", { open: "never" }]],
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
