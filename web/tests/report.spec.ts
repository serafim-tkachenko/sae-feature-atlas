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
const legacy = pathToFileURL(
  resolve("../tmp/browser-fixtures/legacy.html"),
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
  await expect(
    page.getByText("2 texts in this run", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByLabel("Matching features").getByRole("button").first(),
  ).toContainText("Example:");
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

test("context help is keyboard accessible, modal, and returns focus", async ({
  page,
}) => {
  await page.goto(report);
  const trigger = page.getByRole("button", {
    name: "About reading activation contexts",
  });
  await trigger.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", {
    name: "Reading activation contexts",
  });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("it does not encode activation strength");
  await expect(
    dialog.getByRole("button", { name: "Close help" }),
  ).toBeFocused();
  await page.keyboard.press("Tab");
  // Native dialogs may cycle through browser chrome (reported as body).
  // Focus must never reach an interactive element behind the modal.
  expect(
    await page.evaluate(
      () =>
        document.activeElement === document.body ||
        Boolean(document.activeElement?.closest("dialog")),
    ),
  ).toBe(true);
  await page.keyboard.press("Shift+Tab");
  await expect(
    dialog.getByRole("button", { name: "Close help" }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
  await page
    .getByRole("button", {
      name: "About eligible-token frequency",
      exact: true,
    })
    .click();
  await expect(page.getByRole("dialog")).toContainText(
    "1 in 100 eligible tokens",
  );
  await page.getByRole("button", { name: "Close help" }).click();
  await page.goto(legacy);
  await page
    .getByRole("button", { name: "About legacy token frequency" })
    .click();
  await expect(page.getByRole("dialog")).toContainText(
    "denominator and token eligibility policy are not recorded",
  );
});

test("active filters can be removed individually and invalid numeric ranges recover", async ({
  page,
}) => {
  await page.goto(report);
  const list = page.getByLabel("Matching features").getByRole("button");
  await page.getByRole("searchbox").fill("quantum");
  await page.getByLabel("Minimum support", { exact: true }).fill("6");
  await expect(list).toHaveCount(1);
  await expect(page.getByLabel("Active filters")).toContainText(
    "Showing 1 of 32 features",
  );
  await page
    .getByRole("button", { name: "Remove support filter", exact: true })
    .click();
  await expect(list).toHaveCount(2);
  await expect(page.getByRole("searchbox")).toHaveValue("quantum");
  await page.getByLabel("Minimum frequency (%)", { exact: true }).fill("8");
  await page.getByLabel("Maximum frequency (%)", { exact: true }).fill("4");
  await expect(page.getByRole("alert")).toContainText(
    "minimum no greater than maximum",
  );
  await expect(
    page.getByLabel("Minimum frequency (%)", { exact: true }),
  ).toHaveAttribute("aria-invalid", "true");
  await expect(list).toHaveCount(2);
  await page.getByLabel("Maximum frequency (%)", { exact: true }).fill("10");
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(list).toHaveCount(1);
  await page.getByLabel("Minimum support", { exact: true }).fill("-1");
  await expect(page.getByRole("alert")).toContainText(
    "whole number of 0 or more",
  );
  await page
    .getByRole("button", { name: "Reset filters", exact: true })
    .first()
    .click();
  await expect(list).toHaveCount(25);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect(page.getByLabel("Active filters")).toContainText("no filters");
});

test("first-use saving guidance and help fit a narrow viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(report);
  await page.getByRole("button", { name: "Saved (0)", exact: true }).click();
  await expect(page.getByText(/No saved features yet/)).toBeVisible();
  await page
    .getByRole("button", { name: "How saving & sharing works" })
    .click();
  await expect(page.getByRole("dialog")).toContainText(
    "Save at least one feature",
  );
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(375);
  const box = (await page.getByRole("dialog").boundingBox())!;
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(375);
  await page.screenshot({ path: "../tmp/browser-fixtures/mobile-help.png" });
  await page.getByRole("button", { name: "Close help" }).click();
  await page.getByRole("button", { name: "Remove saved filter" }).click();
  await expect(
    page.getByLabel("Matching features").getByRole("button"),
  ).toHaveCount(25);
});

test("feature links focus on evidence, reading modes reveal context, and section navigation moves focus", async ({
  page,
}) => {
  await page.goto(`${report}#feature=7`);
  await expect(page.getByRole("searchbox")).not.toBeVisible();
  const context = page.locator(".context-body > .context-text").first();
  await expect(context).not.toContainText("Earlier saved context.");
  await expect(page.getByRole("meter").first()).toHaveAttribute("max", "3");
  await page.getByRole("button", { name: "Expanded", exact: true }).click();
  await expect(context).toContainText("Earlier saved context.");
  await page.getByRole("button", { name: "Snippet", exact: true }).click();
  await expect(context).not.toContainText("Earlier saved context.");
  await page
    .getByRole("navigation", { name: "Evidence sections" })
    .getByRole("button", { name: "Distribution", exact: true })
    .click();
  await expect(page.locator("#activation-distribution")).toBeFocused();
  await page
    .getByRole("navigation", { name: "Evidence sections" })
    .getByRole("button", { name: /Contexts/ })
    .click();
  await expect(page.locator("#activation-contexts")).toBeFocused();
  await page.locator(".detail-heading").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "../tmp/browser-fixtures/desktop-inspector.png",
  });
});

test("previous and next respect filters, sort order, and the ends of the list", async ({
  page,
}) => {
  await page.goto(report);
  await page.getByRole("searchbox").fill("quantum");
  await expect(
    page.getByRole("button", { name: "Previous feature", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Next feature", exact: true }).click();
  await expect(page).toHaveURL(/#feature=9$/);
  await expect(page.getByRole("searchbox")).toHaveValue("quantum");
  await expect(page.getByLabel("Feature navigation")).toContainText(
    "2 of 2 matching features",
  );
  await expect(
    page.getByRole("button", { name: "Next feature", exact: true }),
  ).toBeDisabled();
  await expect(page.getByRole("meter").first()).toHaveAttribute("max", "2");
  await page
    .getByRole("button", { name: "Previous feature", exact: true })
    .click();
  await expect(page).toHaveURL(/#feature=7$/);
  await page.getByText("Search, filters & landscape", { exact: true }).click();
  await expect(page.getByRole("searchbox")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Remove search filter" }),
  ).toBeVisible();
});

test("saved selections stay in sync across two open report tabs", async ({
  page,
  context,
}) => {
  await page.goto(`${report}#feature=7`);
  const other = await context.newPage();
  await other.goto(`${report}#feature=9`);
  await page.getByRole("button", { name: "Save feature", exact: true }).click();
  await expect(
    other.getByRole("button", { name: "Saved (1)", exact: true }),
  ).toBeVisible();
  await other
    .getByRole("button", { name: "Save feature", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Saved (2)", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Saved", exact: true }).click();
  await expect(
    other.getByRole("button", { name: "Saved (1)", exact: true }),
  ).toBeVisible();
  await other.reload();
  await other.getByRole("button", { name: "Saved (1)", exact: true }).click();
  await expect(
    other.getByLabel("Matching features").getByRole("button"),
  ).toHaveCount(1);
  await expect(
    other.getByLabel("Matching features").getByRole("button"),
  ).toContainText("#9");
});

test("older saved arrays are preserved, deduplicated and can be removed", async ({
  page,
}) => {
  await page.goto(`${report}#feature=7`);
  await page.evaluate(() => {
    const data = JSON.parse(
      document.getElementById("atlas-data")!.textContent!,
    );
    localStorage.setItem(
      `atlas-saved:${data.run.fingerprints.analysis}`,
      JSON.stringify([7, 7, 9, 99999, "7"]),
    );
  });
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Saved (2)", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Saved", exact: true }).click();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Saved (1)", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Save feature", exact: true }),
  ).toBeVisible();
});

test("simultaneous saves in separate tabs retain both features", async ({
  page,
  context,
}) => {
  await page.goto(`${report}#feature=7`);
  const other = await context.newPage();
  await other.goto(`${report}#feature=9`);
  await Promise.all([
    page.getByRole("button", { name: "Save feature", exact: true }).click(),
    other.getByRole("button", { name: "Save feature", exact: true }).click(),
  ]);
  await expect(
    page.getByRole("button", { name: "Saved (2)", exact: true }),
  ).toBeVisible();
  await expect(
    other.getByRole("button", { name: "Saved (2)", exact: true }),
  ).toBeVisible();
});

test("deep links reveal the selected row on its own browser page", async ({
  page,
}) => {
  await page.goto(`${report}#feature=129`);
  await expect(
    page.getByLabel("Matching features").getByRole("button", { name: /#129 / }),
  ).toHaveAttribute("aria-pressed", "true");
});

test("next feature crosses a list page boundary and remains visible", async ({
  page,
}) => {
  await page.goto(`${report}#feature=122`);
  await page.getByRole("button", { name: "Next feature", exact: true }).click();
  await expect(page).toHaveURL(/#feature=123$/);
  await expect(
    page.getByLabel("Matching features").getByRole("button", { name: /#123 / }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Previous", exact: true }).click();
  await expect(
    page.getByLabel("Matching features").getByRole("button"),
  ).toHaveCount(25);
});

test("an empty analysis selection explains recovery and never displays rejected features", async ({
  page,
}) => {
  await page.goto(
    pathToFileURL(resolve("../tmp/browser-fixtures/empty-selection.html")).href,
  );
  await expect(
    page.getByRole("heading", {
      name: "No features in this report",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByLabel("Matching features").getByRole("button"),
  ).toHaveCount(0);
  await page
    .getByText("Run notes · check provenance before comparing results", {
      exact: true,
    })
    .click();
  await expect(
    page.getByText(/No features are present in the saved selection/),
  ).toBeVisible();
});
