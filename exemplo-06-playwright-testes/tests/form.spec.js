import { expect, test } from "@playwright/test";

test.describe("form interactions", () => {
  test("submits a new image card and updates the list", async ({ page }) => {
    await page.goto("/vanilla-js-web-app-example/");

    const title = "Copilot Generated Card";
    const imageUrl = "https://placehold.co/600x400";

    await page.getByRole("textbox", { name: "Image Title" }).fill(title);
    await page.getByRole("textbox", { name: "Image URL" }).fill(imageUrl);
    await page.getByRole("button", { name: "Submit Form" }).click();

    await expect(page.locator("main article")).toHaveCount(4);
    await expect(page.getByRole("heading", { name: title })).toBeVisible();
    await expect(page.locator("main")).toContainText(title);
  });

  test("shows required field validation when submitted empty", async ({ page }) => {
    await page.goto("/vanilla-js-web-app-example/");

    const titleInput = page.getByRole("textbox", { name: "Image Title" });
    const urlInput = page.getByRole("textbox", { name: "Image URL" });

    await page.getByRole("button", { name: "Submit Form" }).click();

    await expect(titleInput.evaluate((input) => input.checkValidity())).resolves.toBe(false);
    await expect(urlInput.evaluate((input) => input.checkValidity())).resolves.toBe(false);
    await expect(page.locator("main article")).toHaveCount(3);
  });

  test("shows url validation when the image url is malformed", async ({ page }) => {
    await page.goto("/vanilla-js-web-app-example/");

    await page.getByRole("textbox", { name: "Image Title" }).fill("Invalid URL Example");
    await page.getByRole("textbox", { name: "Image URL" }).fill("not-a-url");
    await page.getByRole("button", { name: "Submit Form" }).click();

    await expect(page.getByText("Please type a valid URL")).toBeVisible();
    await expect(page.locator("main article")).toHaveCount(3);
  });
});
