const SVG_NS = "http://www.w3.org/2000/svg";
const MODEL = {
  main: "har__mse_log__macro+gvz+slv_iv+us_epu",
  robust: "har__qlike__macro+us_epu",
  persistence: "persistence__raw__none",
};

let state = null;
let predictions = [];
let activeHorizon = 5;

const el = (id) => document.getElementById(id);
const fmt = (value, digits = 2) => Number(value).toFixed(digits);

function svgNode(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function groupFeatures(features) {
  return features.reduce((groups, feature) => {
    (groups[feature.group] ||= []).push(feature);
    return groups;
  }, {});
}

function renderFeatureForm(input) {
  el("input-date").textContent = input.identity.trade_date;
  el("input-contract").textContent = input.identity.selected_au_contract;
  const container = el("feature-groups");
  container.replaceChildren();
  Object.entries(groupFeatures(input.features)).forEach(([group, features], index) => {
    const details = document.createElement("details");
    details.className = "feature-group";
    details.open = index < 2;
    const summary = document.createElement("summary");
    summary.textContent = `${group} · ${features.length}`;
    details.append(summary);
    features.forEach((feature) => {
      const wrapper = document.createElement("div");
      wrapper.className = "feature-field";
      const label = document.createElement("label");
      label.htmlFor = `feature-${feature.name}`;
      label.textContent = feature.label;
      const inputEl = document.createElement("input");
      inputEl.id = `feature-${feature.name}`;
      inputEl.name = feature.name;
      inputEl.type = "number";
      inputEl.step = "any";
      inputEl.required = true;
      inputEl.value = feature.value;
      inputEl.dataset.frozen = feature.value;
      wrapper.append(label, inputEl);
      details.append(wrapper);
    });
    container.append(details);
  });
}

function inputPayload() {
  const payload = {
    trade_date: state.input.identity.trade_date,
    selected_au_contract: state.input.identity.selected_au_contract,
    rv: state.input.identity.rv,
  };
  document.querySelectorAll("#feature-form input").forEach((input) => {
    payload[input.name] = Number(input.value);
  });
  return payload;
}

function drawTermChart(rows) {
  const svg = el("term-chart");
  svg.replaceChildren();
  const width = 760, height = 390;
  const margin = { left: 66, right: 32, top: 34, bottom: 58 };
  const horizons = [5, 20, 40];
  const main = horizons.map((h) => rows.find((r) => r.horizon === h && r.role === "main"));
  const robust = horizons.map((h) => rows.find((r) => r.horizon === h && r.role === "robust"));
  const current = Number(main[0]?.current_vol_pct || 0);
  const values = [...main, ...robust].flatMap((r) => [Number(r.forecast_vol_q10_pct), Number(r.forecast_vol_pct), Number(r.forecast_vol_q90_pct)]).concat(current);
  let min = Math.floor(Math.min(...values) - 2);
  let max = Math.ceil(Math.max(...values) + 2);
  if (max - min < 6) { min -= 2; max += 2; }
  const x = (h) => margin.left + (horizons.indexOf(h) * (width - margin.left - margin.right)) / 2;
  const y = (v) => margin.top + ((max - v) * (height - margin.top - margin.bottom)) / (max - min);
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const value = min + ((max - min) * i) / ticks;
    const yy = y(value);
    svg.append(svgNode("line", { x1: margin.left, y1: yy, x2: width - margin.right, y2: yy, class: "chart-grid" }));
    const label = svgNode("text", { x: margin.left - 12, y: yy + 4, "text-anchor": "end", class: "chart-axis" });
    label.textContent = `${fmt(value, 1)}%`;
    svg.append(label);
  }
  horizons.forEach((h) => {
    const label = svgNode("text", { x: x(h), y: height - 22, "text-anchor": "middle", class: "chart-axis" });
    label.textContent = `${h}D`;
    svg.append(label);
  });
  const lower = main.map((r) => `${x(r.horizon)},${y(Number(r.forecast_vol_q10_pct))}`).join(" ");
  const upper = [...main].reverse().map((r) => `${x(r.horizon)},${y(Number(r.forecast_vol_q90_pct))}`).join(" ");
  svg.append(svgNode("polygon", { points: `${lower} ${upper}`, class: "chart-band" }));
  const path = (series, className) => {
    const points = series.map((r) => `${x(r.horizon)},${y(Number(r.forecast_vol_pct))}`).join(" ");
    svg.append(svgNode("polyline", { points, class: className }));
  };
  path(main, "chart-main");
  path(robust, "chart-robust");
  svg.append(svgNode("line", { x1: margin.left, y1: y(current), x2: width - margin.right, y2: y(current), class: "chart-current" }));
  [...main.map((r) => [r, "chart-point-main"]), ...robust.map((r) => [r, "chart-point-robust"])].forEach(([row, cls]) => {
    svg.append(svgNode("circle", { cx: x(row.horizon), cy: y(Number(row.forecast_vol_pct)), r: 7, class: cls }));
  });
  const currentLabel = svgNode("text", { x: width - margin.right, y: y(current) - 8, "text-anchor": "end", class: "chart-axis" });
  currentLabel.textContent = `CURRENT ${fmt(current)}%`;
  svg.append(currentLabel);
}

function renderReadouts(rows) {
  const box = el("forecast-readouts");
  box.replaceChildren();
  [5, 20, 40].forEach((horizon) => {
    const main = rows.find((r) => r.horizon === horizon && r.role === "main");
    const robust = rows.find((r) => r.horizon === horizon && r.role === "robust");
    const item = document.createElement("div");
    item.className = "readout";
    item.innerHTML = `<span>${horizon} 个交易日</span><div class="readout-values"><b class="main">${fmt(main.forecast_vol_pct)}%</b><b class="robust">${fmt(robust.forecast_vol_pct)}%</b></div>`;
    box.append(item);
  });
}

function renderDiagnostics(rows) {
  const current = Number(rows[0]?.current_vol_pct || 0);
  const spreads = rows.filter((r) => r.role === "main").map((r) => Number(r.model_spread_vol_points));
  const agreements = new Set(rows.map((r) => r.model_agreement));
  const agreed = agreements.size === 1 && agreements.has("AGREE");
  el("current-vol").textContent = `${fmt(current)}%`;
  el("max-spread").textContent = `${fmt(Math.max(...spreads))} pt`;
  el("agreement-label").textContent = agreed ? "方向一致" : "存在分歧";
  el("agreement-detail").textContent = agreed ? "三种期限的双模型方向通过" : "至少一个期限需要人工复核";
  document.querySelector(".assay-stamp").classList.toggle("disagree", !agreed);
  const allUp = rows.every((r) => r.direction_vs_current === "UP");
  const allDown = rows.every((r) => r.direction_vs_current === "DOWN");
  el("interpretation").textContent = allUp
    ? "两套模型在全部期限都预计波动率高于当前 RV。进入策略层后仍须比较期权 IV 与含 VRP 的公平波动率。"
    : allDown
      ? "两套模型在全部期限都预计波动率低于当前 RV。只有当市场 IV 相对公平波动率仍足够昂贵，才可能形成卖波动机会。"
      : "期限结构方向并不统一。应逐期限匹配期权到期日，并对模型分歧较大的期限降低信号权重或跳过。";
}

function renderPredictions(rows) {
  predictions = rows.map((row) => ({ ...row, horizon: Number(row.horizon) }));
  el("latest-date").textContent = predictions[0]?.forecast_date || "—";
  drawTermChart(predictions);
  renderReadouts(predictions);
  renderDiagnostics(predictions);
}

function linePoints(rows, valueKey, x, y) {
  return rows.map((row) => `${x(new Date(row.trade_date))},${y(Number(row[valueKey]))}`).join(" ");
}

function drawHistory() {
  const svg = el("history-chart");
  svg.replaceChildren();
  const width = 1080, height = 430;
  const margin = { left: 62, right: 24, top: 32, bottom: 46 };
  const all = state.history.filter((r) => Number(r.horizon) === activeHorizon);
  const main = all.filter((r) => r.model_name === MODEL.main);
  const robust = all.filter((r) => r.model_name === MODEL.robust);
  if (!main.length || !robust.length) {
    const notice = svgNode("text", { x: width / 2, y: height / 2, "text-anchor": "middle", class: "chart-axis" });
    notice.textContent = "Public portfolio excludes licensed walk-forward history; aggregate metrics remain available below.";
    svg.append(notice);
    return;
  }
  const dates = main.map((r) => new Date(r.trade_date));
  const minDate = Math.min(...dates), maxDate = Math.max(...dates);
  const values = [...main.flatMap((r) => [r.forecast_vol_pct, r.actual_vol_pct]), ...robust.map((r) => r.forecast_vol_pct)].map(Number);
  let min = Math.max(0, Math.floor(Math.min(...values) - 3));
  let max = Math.ceil(Math.max(...values) + 3);
  const x = (date) => margin.left + ((date - minDate) * (width - margin.left - margin.right)) / (maxDate - minDate || 1);
  const y = (value) => margin.top + ((max - value) * (height - margin.top - margin.bottom)) / (max - min || 1);
  for (let i = 0; i <= 5; i++) {
    const value = min + ((max - min) * i) / 5;
    const yy = y(value);
    svg.append(svgNode("line", { x1: margin.left, y1: yy, x2: width - margin.right, y2: yy, class: "chart-grid" }));
    const label = svgNode("text", { x: margin.left - 10, y: yy + 4, "text-anchor": "end", class: "chart-axis" });
    label.textContent = `${fmt(value, 0)}%`;
    svg.append(label);
  }
  [0, .25, .5, .75, 1].forEach((ratio) => {
    const date = new Date(minDate + (maxDate - minDate) * ratio);
    const label = svgNode("text", { x: x(date), y: height - 15, "text-anchor": "middle", class: "chart-axis" });
    label.textContent = date.toISOString().slice(0, 7);
    svg.append(label);
  });
  svg.append(svgNode("polyline", { points: linePoints(main, "actual_vol_pct", x, y), class: "chart-actual" }));
  svg.append(svgNode("polyline", { points: linePoints(main, "forecast_vol_pct", x, y), class: "chart-main" }));
  svg.append(svgNode("polyline", { points: linePoints(robust, "forecast_vol_pct", x, y), class: "chart-robust" }));
  const legend = [
    ["实现", "#102b4c", 805], ["主模型", "#aa7b19", 880], ["稳健", "#287565", 970],
  ];
  legend.forEach(([name, color, xx]) => {
    svg.append(svgNode("line", { x1: xx, y1: 16, x2: xx + 24, y2: 16, stroke: color, "stroke-width": 3 }));
    const label = svgNode("text", { x: xx + 30, y: 20, class: "chart-axis" });
    label.textContent = name;
    svg.append(label);
  });
}

function modelLabel(name) {
  if (name === MODEL.main) return "主模型 HAR-X";
  if (name === MODEL.robust) return "无 IV 稳健 HAR-X";
  return "Persistence";
}

function renderMetrics(metrics) {
  const body = el("metrics-body");
  body.replaceChildren();
  metrics.forEach((row) => {
    const tr = document.createElement("tr");
    tr.className = row.model_name === MODEL.main ? "model-main" : row.model_name === MODEL.robust ? "model-robust" : "model-persistence";
    tr.innerHTML = `<td>${row.horizon} 日</td><td>${modelLabel(row.model_name)}</td><td>${row.sample_count}</td><td>${fmt(row.qlike, 3)}</td><td>${fmt(row.rmse_log_rv, 3)}</td><td>${fmt(100 * row.oos_r2_vs_persistence, 1)}%</td><td>${fmt(100 * row.direction_accuracy, 1)}%</td>`;
    body.append(tr);
  });
}

async function runPrediction(event) {
  event.preventDefault();
  const status = el("form-status");
  status.classList.remove("error");
  status.textContent = "模型计算中…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: inputPayload() }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "模型调用失败");
    renderPredictions(payload.predictions);
    status.textContent = `完成：${new Date().toLocaleTimeString("zh-CN")}`;
  } catch (error) {
    status.classList.add("error");
    status.textContent = `无法预测：${error.message}`;
  }
}

function resetInputs() {
  document.querySelectorAll("#feature-form input").forEach((input) => { input.value = input.dataset.frozen; });
  renderPredictions(state.reference);
  el("form-status").textContent = "已恢复冻结样本与参考预测。";
}

async function boot() {
  try {
    const response = await fetch("/api/state");
    if (!response.ok) throw new Error("无法读取模型状态");
    state = await response.json();
    el("model-version").textContent = state.manifest.model_package_version;
    renderFeatureForm(state.input);
    renderPredictions(state.reference);
    renderMetrics(state.metrics);
    drawHistory();
    el("feature-form").addEventListener("submit", runPrediction);
    el("reset-button").addEventListener("click", resetInputs);
    document.querySelectorAll(".horizon-tab").forEach((button) => {
      button.addEventListener("click", () => {
        activeHorizon = Number(button.dataset.horizon);
        document.querySelectorAll(".horizon-tab").forEach((item) => item.classList.toggle("active", item === button));
        drawHistory();
      });
    });
    el("form-status").textContent = "冻结样本已载入，可直接运行或修改输入。";
  } catch (error) {
    el("form-status").classList.add("error");
    el("form-status").textContent = error.message;
  }
}

boot();
