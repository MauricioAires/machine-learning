import { expect, test } from "@playwright/test";

test("loads the gallery page", async ({ page }) => {
  await page.goto("/vanilla-js-web-app-example/");

  await expect(page).toHaveTitle("TDD Frontend Example");
  await expect(page.locator("main article")).toHaveCount(3);
  await expect(page.locator("main")).toContainText("AI Alien");
  await expect(page.locator("main")).toContainText("Predator Night Vision");
  await expect(page.locator("main")).toContainText("ET Bilu");
});
