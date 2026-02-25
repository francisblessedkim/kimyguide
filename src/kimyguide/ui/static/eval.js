let barChart = null;
let lineChart = null;

function safeNum(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}

// Supports either {mean, lo, hi} OR {mean, low, high}
function isCIObj(x) {
  if (!x || typeof x !== "object") return false;
  const hasMean = "mean" in x;
  const hasLoHi = "lo" in x && "hi" in x;
  const hasLowHigh = "low" in x && "high" in x;
  return hasMean && (hasLoHi || hasLowHigh);
}

function ciMean(x) {
  return isCIObj(x) ? safeNum(x.mean) : safeNum(x);
}

function ciLo(x) {
  if (!isCIObj(x)) return null;
  return "lo" in x ? safeNum(x.lo) : safeNum(x.low);
}

function ciHi(x) {
  if (!isCIObj(x)) return null;
  return "hi" in x ? safeNum(x.hi) : safeNum(x.high);
}

function fmtCI(x) {
  if (!isCIObj(x)) return safeNum(x).toFixed(4);
  const lo = ciLo(x);
  const hi = ciHi(x);
  return `${safeNum(x.mean).toFixed(4)} (${lo.toFixed(4)}–${hi.toFixed(4)})`;
}

function renderTable(rows) {
  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";

  for (const r of rows) {
    const model = r.model ?? r.name ?? "model";

    const mrr = r.mrr;
    const recall = r.recall;
    const ndcg = r.ndcg;

    const cov = r.coverage;
    const div = r.diversity;

    const tr = document.createElement("tr");
    tr.className = "border-b border-slate-100 hover:bg-blue-50/40";

    tr.innerHTML = `
      <td class="py-3 font-medium">${model}</td>
      <td class="py-3">${fmtCI(mrr)}</td>
      <td class="py-3">${fmtCI(recall)}</td>
      <td class="py-3">${fmtCI(ndcg)}</td>
      <td class="py-3">${cov == null ? "-" : safeNum(cov).toFixed(4)}</td>
      <td class="py-3">${div == null ? "-" : safeNum(div).toFixed(4)}</td>
    `;

    tbody.appendChild(tr);
  }
}

function renderBar(rows) {
  const canvas = document.getElementById("barMetrics");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  const labels = rows.map((r) => r.model ?? r.name ?? "model");
  const mrr = rows.map((r) => ciMean(r.mrr));
  const recall = rows.map((r) => ciMean(r.recall));
  const ndcg = rows.map((r) => ciMean(r.ndcg));

  if (barChart) barChart.destroy();

  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "MRR@K", data: mrr },
        { label: "Recall@K", data: recall },
        { label: "nDCG@K", data: ndcg },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "top" },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        y: { beginAtZero: true, suggestedMax: 1 },
      },
    },
  });
}

function renderSweep(sweep, metric) {
  // expected:
  // { k:[...], series:[{model:"tfidf", mrr:[...], recall:[...], ndcg:[...]}, ...] }
  if (!sweep || !Array.isArray(sweep.k) || !Array.isArray(sweep.series)) return;

  const canvas = document.getElementById("lineSweep");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (lineChart) lineChart.destroy();

  const datasets = [];
  for (const s of sweep.series) {
    const arr = Array.isArray(s[metric]) ? s[metric] : [];
    datasets.push({
      label: `${s.model} ${metric.toUpperCase()}`,
      data: arr.map(safeNum),
      tension: 0.25,
    });
  }

  lineChart = new Chart(ctx, {
    type: "line",
    data: { labels: sweep.k, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top" } },
      scales: { y: { beginAtZero: true, suggestedMax: 1 } },
    },
  });
}

function setMeta(meta, nq, k) {
  const metaEl = document.getElementById("meta");
  if (!metaEl) return;

  const mode = meta?.mode ? `Mode: ${meta.mode}` : "";
  const note = meta?.note ? ` • ${meta.note}` : "";
  const models = meta?.models ? ` • Models: ${meta.models.join(", ")}` : "";

  const base = meta
    ? `Queries: ${meta.n_queries} • K: ${meta.k} • Dataset size: ${meta.num_items ?? "-"}`
    : `Queries: ${nq} • K: ${k}`;

  metaEl.textContent = `${base}${models}${mode ? " • " + mode : ""}${note}`;
}

async function runEval() {
  const nq = Number(document.getElementById("nq")?.value || 60);
  const k = Number(document.getElementById("k")?.value || 10);
  const mode = String(document.getElementById("mode")?.value || "subject").trim();
  const metric = String(document.getElementById("sweepMetric")?.value || "mrr").trim();

  const status = document.getElementById("status");
  if (status) status.textContent = "Running evaluation…";

  const url = `/api/eval?nq=${encodeURIComponent(nq)}&k=${encodeURIComponent(k)}&mode=${encodeURIComponent(
    mode
  )}&n_boot=200`;

  const res = await fetch(url, { headers: { Accept: "application/json" } });

  if (!res.ok) {
    const txt = await res.text();
    if (status) status.textContent = `Failed: ${res.status} ${txt}`;
    return;
  }

  const data = await res.json();

  setMeta(data?.meta, nq, k);

  const rows = data?.rows || [];
  if (!rows.length) {
    if (status) status.textContent = "No results returned.";
    return;
  }

  renderTable(rows);
  renderBar(rows);
  renderSweep(data?.sweep, metric);

  if (status) status.textContent = "Done.";
}

document.getElementById("run")?.addEventListener("click", runEval);

document.getElementById("sweepMetric")?.addEventListener("change", () => {
  // simplest: rerun to refresh chart
  runEval();
});