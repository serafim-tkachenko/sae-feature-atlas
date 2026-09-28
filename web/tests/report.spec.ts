import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const report = pathToFileURL(
  resolve("../tmp/browser-fixtures/report.html"),
).href;
const partial = pathToFileURL(
  resolve("../tmp/browser-fixtures/partial.html"),
).href;

test("opens directly from disk offline, with literal context text and real histogram", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const external: string[] = [];
  page.on("request", (request) => {
    if (/^https?:/.test(request.url())) external.push(request.url());
  });
  await page.goto(report);
  await expect(
    page.getByRole("heading", { name: "Explore features" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Feature 7", exact: true }),
  ).toBeVisible();
  await expect(page.locator("mark").first()).toHaveText(
    "<script>globalThis.injected=true</script>",
  );
  expect(
    await page.evaluate(() => (globalThis as any).injected),
  ).toBeUndefined();
  await expect(
    page.getByRole("img", { name: /Histogram of 5 stored/ }),
  ).toBeVisible();
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
});

test("search, numeric filters, empty state, reset and pagination", async ({
  page,
}) => {
  await page.goto(report);
  const list = page.getByLabel("Matching features");
  await expect(list.getByRole("button")).toHaveCount(25);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(list.getByRole("button")).toHaveCount(7);
  await page.getByRole("searchbox").fill("quantum");
  await expect(list.getByRole("button")).toHaveCount(2);
  await page.getByLabel("Minimum support", { exact: true }).fill("6");
  await expect(list.getByRole("button")).toHaveCount(1);
  await page.getByLabel("Maximum frequency (%)", { exact: true }).fill("1");
  await expect(
    page.getByText("No features match these filters."),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Reset filters", exact: true })
    .first()
    .click();
  await expect(list.getByRole("button")).toHaveCount(25);
  await page.getByRole("searchbox").fill("7");
  await expect(list.getByRole("button")).toHaveCount(1);
  await expect(list.getByRole("button")).toContainText("#7");
});

test("neighbors navigate, deep links survive reload, and browser back restores feature", async ({
  page,
}) => {
  await page.goto(`${report}#feature=7`);
  await page.getByRole("button", { name: "9 ↗", exact: true }).first().click();
  await expect(page).toHaveURL(/#feature=9$/);
  await expect(
    page.getByRole("heading", { name: "Feature 9", exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Feature 9", exact: true }),
  ).toBeVisible();
  await page.goBack();
  await expect(
    page.getByRole("heading", { name: "Feature 7", exact: true }),
  ).toBeVisible();
});

test("save, restore, filter and export evidence with provenance", async ({
  page,
}) => {
  await page.goto(report);
  await page.getByRole("button", { name: "Save feature", exact: true }).click();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Saved (1)", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Saved (1)", exact: true }).click();
  await expect(
    page.getByLabel("Matching features").getByRole("button"),
  ).toHaveCount(1);
  const downloadPromise = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export selection", exact: true })
    .click();
  const download = await downloadPromise;
  const exported = JSON.parse(readFileSync((await download.path())!, "utf8"));
  expect(exported.features.map((f: any) => f.id)).toEqual([7]);
  expect(exported.run.fingerprints.analysis).toBeTruthy();
  expect(exported.features[0].examples.top[0].evidence_id).toContain("f7:");
  expect(exported.features[0].histogram.population).toBe(
    "all_stored_activations",
  );
});

test("chart brushing filters the list and can be cleared", async ({ page }) => {
  await page.goto(report);
  const chart = page.getByRole("img", { name: /Feature frequency versus/ });
  const box = (await chart.boundingBox())!;
  await page.mouse.move(box.x + 65, box.y + 18);
  await page.mouse.down();
  await page.mouse.move(box.x + 110, box.y + 60, { steps: 5 });
  await page.mouse.up();
  await expect(
    page.getByLabel("Matching features").getByRole("button"),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Clear chart selection" }).click();
  await expect(
    page.getByLabel("Matching features").getByRole("button"),
  ).toHaveCount(25);
});

test("partial legacy runs distinguish corrupt, empty and missing evidence", async ({
  page,
}) => {
  await page.goto(partial);
  await expect(page.getByText("Unverified / legacy run")).toBeVisible();
  await page.getByRole("button", { name: "Run details", exact: true }).click();
  const details = page.getByRole("region", {
    name: "Run details",
    exact: true,
  });
  await expect(
    details.getByRole("row").filter({ hasText: "feature_cards.parquet" }),
  ).toContainText("unreadable");
  await expect(
    details.getByRole("row").filter({ hasText: "decoder_neighbors.parquet" }),
  ).toContainText("empty");
  await expect(
    details.getByRole("row").filter({ hasText: "coactivation_pairs.parquet" }),
  ).toContainText("missing");
});

test("narrow viewport stays readable without page overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(report);
  await expect(
    page.getByRole("heading", { name: "Feature 7", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(375);
  await page.getByRole("button", { name: "Run details", exact: true }).click();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(375);
  await page.screenshot({
    path: "../tmp/browser-fixtures/mobile.png",
    fullPage: true,
  });
});

test("unknown feature links and absent regime examples are explicit", async ({
  page,
}) => {
  await page.goto(`${report}#feature=99999`);
  await expect(
    page.getByRole("heading", { name: "Feature 99999 is not in this report" }),
  ).toBeVisible();
  await page
    .getByLabel("Matching features")
    .getByRole("button")
    .first()
    .click();
  await page.getByRole("button", { name: "Low regime 0" }).click();
  await expect(page.getByText(/No low-regime contexts saved/)).toBeVisible();
});
