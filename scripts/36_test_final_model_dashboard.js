#!/usr/bin/env node
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const QA = path.join(ROOT, "deliverables", "qa", "final_model_dashboard");
fs.mkdirSync(QA, { recursive: true });

(async () => {
  const consoleErrors = [];
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 }, deviceScaleFactor: 1 });
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  await page.goto("http://127.0.0.1:8765", { waitUntil: "networkidle" });
  await page.waitForSelector("#forecast-readouts .readout");
  if ((await page.title()) !== "沪金波动率｜模型观测台") throw new Error("Unexpected page title");
  if ((await page.locator("#forecast-readouts .readout").count()) !== 3) throw new Error("Expected three horizon readouts");
  if ((await page.locator("#metrics-body tr").count()) !== 9) throw new Error("Expected nine metric rows");
  if ((await page.locator("#feature-form input").count()) !== 15) throw new Error("Expected fifteen model inputs");
  if ((await page.locator("#agreement-label").innerText()) !== "方向一致") throw new Error("Reference models should agree");
  await page.locator('[data-horizon="20"]').click();
  if (!((await page.locator('[data-horizon="20"]').getAttribute("class")) || "").includes("active")) throw new Error("Horizon tab did not activate");
  const first = page.locator("#feature-form input").first();
  await first.fill(String(Number(await first.inputValue()) * 1.01));
  await page.getByRole("button", { name: "运行 5 / 20 / 40 日预测" }).click();
  await page.waitForFunction(() => document.querySelector("#form-status").textContent.startsWith("完成："));
  if ((await page.locator("#form-status").innerText()).includes("无法预测")) throw new Error("Prediction form failed");
  await page.screenshot({ path: path.join(QA, "dashboard_desktop.png"), fullPage: true });

  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  mobile.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(`mobile: ${message.text()}`);
  });
  await mobile.goto("http://127.0.0.1:8765", { waitUntil: "networkidle" });
  await mobile.waitForSelector("#forecast-readouts .readout");
  await mobile.screenshot({ path: path.join(QA, "dashboard_mobile.png"), fullPage: true });
  await browser.close();

  const report = {
    desktop_screenshot: path.join(QA, "dashboard_desktop.png"),
    mobile_screenshot: path.join(QA, "dashboard_mobile.png"),
    console_errors: consoleErrors,
  };
  fs.writeFileSync(path.join(QA, "browser_test.json"), JSON.stringify(report, null, 2));
  if (consoleErrors.length) throw new Error(`Browser console errors: ${consoleErrors.join(" | ")}`);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
})().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
