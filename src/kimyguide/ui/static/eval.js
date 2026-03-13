let barChart = null;
let lineChart = null;
let lastEvalData = null;
let overlayTimer = null;
let loadingMessageTimer = null;

function safeNum(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}

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

function showOverlay() {
  const overlay = document.getElementById("evalOverlay");
  const inlineLoader = document.getElementById("inlineLoader");
  const overlayText = document.getElementById("overlayText");

  inlineLoader?.classList.remove("hidden");
  inlineLoader?.classList.add("flex");

  overlay?.classList.remove("hidden");

  const messages = [
    "Sampling evaluation queries from the dataset.",
    "Running recommendation models and collecting rankings.",
    "Computing MRR, Recall, nDCG, coverage, and diversity.",
    "Preparing charts and summary metrics for display."
  ];

  let idx = 0;
  overlayText.textContent = messages[idx];

  loadingMessageTimer = setInterval(() => {
    idx = (idx + 1) % messages.length;
    overlayText.textContent = messages[idx];
  }, 1800);
}

function hideOverlay() {
  const overlay = document.getElementById("evalOverlay");
  const inlineLoader = document.getElementById("inlineLoader");

  overlay?.classList.add("hidden");
  inlineLoader?.classList.add("hidden");
  inlineLoader?.classList.remove("flex");

  if (loadingMessageTimer) {
    clearInterval(loadingMessageTimer);
    loadingMessageTimer = null;
  }
  if (overlayTimer) {
    clearTimeout(overlayTimer);
    overlayTimer = null;
  }
}

function renderTable(rows) {
  const tbody = document.getElementById("tbody");
  tbody.innerHTML = "";

  for (const r of rows) {
    const model = r.model ?? r.name ?? "model";

    const tr = document.createElement("tr");
    tr.className = "border-b border-slate-100 hover:bg-blue-50/40 transition";

    tr.innerHTML = `
      <td class="py-4 px-4 font-medium capitalize">${model}</td>
      <td class="py-4 px-4">${fmtCI(r.mrr)}</td>
      <td class="py-4 px-4">${fmtCI(r.recall)}</td>
      <td class="py-4 px-4">${fmtCI(r.ndcg)}</td>
      <td class="py-4 px-4">${r.coverage == null ? "-" : safeNum(r.coverage).toFixed(4)}</td>
      <td class="py-4 px-4">${r.diversity == null ? "-" : safeNum(r.diversity).toFixed(4)}</td>
    `;

    tbody.appendChild(tr);
  }
}

function renderBar(rows) {
  const canvas = document.getElementById("barMetrics");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const labels = rows.map((r) => (r.model ?? r.name ?? "model").toUpperCase());
  const mrr = rows.map((r) => ciMean(r.mrr));
  const recall = rows.map((r) => ciMean(r.recall));
  const ndcg = rows.map((r) => ciMean(r.ndcg));

  if (barChart) barChart.destroy();

  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "MRR@K", data: mrr, borderRadius: 8 },
        { label: "Recall@K", data: recall, borderRadius: 8 },
        { label: "nDCG@K", data: ndcg, borderRadius: 8 },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 900 },
      plugins: {
        legend: { position: "top" },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        y: { beginAtZero: true, suggestedMax: 1 }
      }
    }
  });
}

function renderSweep(sweep, metric) {
  if (!sweep || !Array.isArray(sweep.k) || !Array.isArray(sweep.series)) return;

  const canvas = document.getElementById("lineSweep");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (lineChart) lineChart.destroy();

  const datasets = sweep.series.map((s) => ({
    label: `${s.model} ${metric.toUpperCase()}`,
    data: (Array.isArray(s[metric]) ? s[metric] : []).map(safeNum),
    tension: 0.32,
    fill: false,
    pointRadius: 3,
    pointHoverRadius: 5,
  }));

  lineChart = new Chart(ctx, {
    type: "line",
    data: { labels: sweep.k, datasets },
    options: {
      responsive: true,
      animation: { duration: 900 },
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top" } },
      scales: { y: { beginAtZero: true, suggestedMax: 1 } },
    },
  });
}

function setMeta(meta, nq, k) {
  const metaEl = document.getElementById("meta");
  const statQueries = document.getElementById("statQueries");
  const statK = document.getElementById("statK");
  const statDataset = document.getElementById("statDataset");
  const statMode = document.getElementById("statMode");

  const resolvedQueries = meta && meta.n_queries != null ? meta.n_queries : nq;
  const resolvedK = meta && meta.k != null ? meta.k : k;
  const resolvedDataset = meta && meta.num_items != null ? meta.num_items : "—";
  const resolvedMode = meta && meta.mode ? meta.mode : "—";

  const note = meta?.note ? ` • ${meta.note}` : "";
  const models = meta?.models ? ` • Models: ${meta.models.join(", ")}` : "";

  if (metaEl) {
    metaEl.textContent =
      `Queries: ${resolvedQueries} • K: ${resolvedK} • Dataset size: ${resolvedDataset}${models} • Mode: ${resolvedMode}${note}`;
  }

  if (statQueries) statQueries.textContent = String(resolvedQueries);
  if (statK) statK.textContent = String(resolvedK);
  if (statDataset) statDataset.textContent = String(resolvedDataset);
  if (statMode) statMode.textContent = String(resolvedMode);
}

async function runEval() {
  const nq = Number(document.getElementById("nq")?.value || 60);
  const k = Number(document.getElementById("k")?.value || 10);
  const mode = String(document.getElementById("mode")?.value || "subject").trim();
  const metric = String(document.getElementById("sweepMetric")?.value || "mrr").trim();
  const status = document.getElementById("status");
  // set the cards
  const statQueries = document.getElementById("statQueries");
  const statK = document.getElementById("statK");
  const statDataset = document.getElementById("statDataset");
  const statMode = document.getElementById("statMode");

  if (statQueries) statQueries.textContent = "…";
  if (statK) statK.textContent = "…";
  if (statDataset) statDataset.textContent = "…";
  if (statMode) statMode.textContent = "…";

  if (status) {
    status.textContent = "Running evaluation…";
    status.className = "text-sm text-slate-500";
  }

  overlayTimer = setTimeout(showOverlay, 250);

  try {
    const url = `/api/eval?nq=${encodeURIComponent(nq)}&k=${encodeURIComponent(k)}&mode=${encodeURIComponent(mode)}&n_boot=200`;
    const res = await fetch(url, { headers: { Accept: "application/json" } });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`${res.status} ${txt}`);
    }

    const data = await res.json();
    lastEvalData = data;

    setMeta(data?.meta, nq, k);
    requestAnimationFrame(() => setMeta(data?.meta, nq, k));

    const rows = data?.rows || [];
    if (!rows.length) {
      if (status) {
        status.textContent = "No evaluation results were returned.";
        status.className = "text-sm text-amber-700";
      }
      return;
    }

    renderTable(rows);
    renderBar(rows);
    renderSweep(data?.sweep, metric);

    if (status) {
      status.textContent = "Evaluation complete.";
      status.className = "text-sm text-emerald-700";
    }
  } catch (err) {
    console.error(err);
    if (status) {
      status.textContent = `Failed to run evaluation: ${err.message}`;
      status.className = "text-sm text-red-600";
    }
  } finally {
    hideOverlay();
  }
}

document.getElementById("run")?.addEventListener("click", runEval);

document.getElementById("sweepMetric")?.addEventListener("change", () => {
  const metric = String(document.getElementById("sweepMetric")?.value || "mrr").trim();
  if (lastEvalData?.sweep) {
    renderSweep(lastEvalData.sweep, metric);
  }
});