/* miniLLM Admin Console — 原生 JS，无构建 */
"use strict";

const $ = (id) => document.getElementById(id);
let refreshInterval = 5000;
let engineData = [];

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmt(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return Number(v).toFixed(digits);
}

/* ---------------- 引擎状态 ---------------- */

function pickMetric(m, ...keys) {
  for (const k of keys) {
    if (m[k] !== undefined) return m[k];
  }
  return null;
}

function engineCard(e) {
  const m = e.metrics || {};
  const tps = pickMetric(m, "minillm_tokens_generated_total") ? e.tokens_per_sec : null;
  const kv = [
    ["请求总数", fmt(pickMetric(m, "minillm_requests_total", "vllm:num_requests_total"), 0)],
    ["生成 tokens", fmt(pickMetric(m, "minillm_tokens_generated_total", "vllm:generation_tokens_total"), 0)],
    ["tokens/s", tps !== null ? fmt(tps) : "-"],
    ["TTFT 样本数", fmt(pickMetric(m, "minillm_ttft_seconds_count", "vllm:time_to_first_token_seconds_count"), 0)],
    ["运行中请求", fmt(pickMetric(m, "vllm:num_requests_running"), 0)],
    ["排队请求", fmt(pickMetric(m, "vllm:num_requests_waiting"), 0)],
    ["Cache 命中率", fmt(pickMetric(m, "vllm:cache_hit_rate"), 2)],
    ["GPU Cache 使用率", fmt(pickMetric(m, "vllm:gpu_cache_usage_perc"), 2)],
  ];
  return `
  <div class="card">
    <div class="head">
      <span class="name">${esc(e.name)} <span style="color:var(--muted)">(${esc(e.type)})</span></span>
      <span><span class="dot ${e.up ? "up" : "down"}"></span>${e.up ? "在线" : "离线"}</span>
    </div>
    <div class="model">${esc(e.model || "—")}</div>
    <div class="kv">${kv.map(([k, v]) => `<span class="k">${k}</span><span class="v ${k === "tokens/s" ? "big" : ""}">${v}</span>`).join("")}</div>
    ${e.error ? `<div class="err">⚠ ${esc(e.error)}</div>` : ""}
  </div>`;
}

async function refreshStatus() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    engineData = data.engines || [];
    refreshInterval = (data.refresh_interval_s || 5) * 1000;
    $("serve-name").textContent = data.serve_name;
    $("refresh-pill").className = "status-pill ok";
    $("refresh-pill").textContent = `● 已连接 · ${new Date().toLocaleTimeString()}`;
    $("engine-cards").innerHTML = engineData.length
      ? engineData.map(engineCard).join("")
      : '<div class="empty">未配置引擎（configs/web/admin.yaml → engines）</div>';
    renderGpu(data.prometheus_url);
  } catch (err) {
    $("refresh-pill").className = "status-pill";
    $("refresh-pill").textContent = `● 连接失败: ${esc(err.message)}`;
  }
}

/* ---------------- GPU（经 Prometheus 代理） ---------------- */

async function renderGpu(prometheusUrl) {
  $("gpu-section").hidden = !prometheusUrl;
  if (!prometheusUrl) return;
  const queries = {
    "GPU 利用率 %": 'DCGM_FI_DEV_GPU_UTIL',
    "GPU 显存已用 GB": 'DCGM_FI_DEV_FB_USED / 1024^3',
    "GPU 温度 ℃": 'DCGM_FI_DEV_GPU_TEMP',
  };
  const cards = [];
  for (const [label, q] of Object.entries(queries)) {
    try {
      const resp = await fetch(`/api/prometheus?query=${encodeURIComponent(q)}`);
      const data = await resp.json();
      const results = (data.data && data.data.result) || [];
      const values = results.map((r) => [r.metric.uuid || "gpu", Number(r.value[1])]);
      cards.push(`<div class="card"><div class="head"><span class="name">${label}</span></div>
        <div class="kv">${values.map(([u, v]) => `<span class="k">${esc(u)}</span><span class="v big">${fmt(v)}</span>`).join("")}</div></div>`);
    } catch { /* prometheus 未就绪时静默 */ }
  }
  $("gpu-cards").innerHTML = cards.join("") || '<div class="empty">Prometheus 未返回数据</div>';
}

/* ---------------- Benchmark ---------------- */

async function loadBench() {
  const resp = await fetch("/api/bench");
  const { runs } = await resp.json();
  if (!runs.length) return;
  const rows = runs.map((r, i) => {
    const best = Math.max(...r.metrics.map((m) => m.throughput_tps || 0), 0.001);
    const bars = [...new Set(r.metrics.map((m) => m.engine))]
      .map((engine) => {
        const tps = Math.max(...r.metrics.filter((m) => m.engine === engine).map((m) => m.throughput_tps || 0));
        return `<div class="bar-row"><span class="label">${esc(engine)}</span>
          <div class="track"><div class="fill" style="width:${(tps / best) * 100}%"></div></div>
          <span class="val">${fmt(tps)} t/s</span></div>`;
      }).join("");
    const findings = (r.findings || []).map((f) =>
      `<span class="finding ${esc(f.severity)}">${esc(f.message)}</span>`).join("");
    return `<tr class="clickable" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
      <td class="mono">${esc(r.experiment)}</td><td>${esc(r.timestamp || "-")}</td>
      <td class="mono">${r.n_cases}</td><td>${findings || "-"}</td></tr>
      <tr class="detail"><td colspan="4">${bars}<br>${esc(r.dir)}</td></tr>`;
  }).join("");
  $("bench-list").innerHTML = `<table><thead><tr>
    <th>实验</th><th>时间</th><th>用例数</th><th>瓶颈分析</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/* ---------------- 训练 ---------------- */

async function loadTrain() {
  const resp = await fetch("/api/train");
  const { runs } = await resp.json();
  if (!runs.length) return;
  const rows = runs.map((r) => `<tr>
    <td class="mono">${esc(r.experiment)}</td>
    <td>${esc(r.finetune_mode)}</td>
    <td class="mono" style="word-break:break-all">${esc(r.model || "-")}</td>
    <td>${esc(r.device || "-")}</td>
    <td class="mono">${r.eval_loss !== undefined && r.eval_loss !== null ? fmt(r.eval_loss, 4) : "-"}</td>
    <td class="mono">${fmt(r.train_tokens_per_second, 1)}</td>
  </tr>`).join("");
  $("train-list").innerHTML = `<table><thead><tr>
    <th>实验</th><th>模式</th><th>模型</th><th>设备</th><th>eval loss</th><th>训练 tokens/s</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

/* ---------------- 启动 ---------------- */

$("btn-refresh").addEventListener("click", () => { refreshStatus(); loadBench(); loadTrain(); });

async function boot() {
  await Promise.all([refreshStatus(), loadBench(), loadTrain()]);
  setInterval(refreshStatus, refreshInterval);
}
boot();
