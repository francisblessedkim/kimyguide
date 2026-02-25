function esc(s) {
  return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function card(rec) {
  const meta = rec.metadata || {};
  const url = meta.url || "";
  return `
    <div class="bg-white border border-slate-200 rounded-2xl shadow-sm p-4">
      <div class="font-semibold tracking-tight">${esc(rec.title)}</div>
      <div class="text-xs text-slate-500 mt-1">score: ${Number(rec.score).toFixed(3)}</div>
      ${url ? `<a class="text-xs text-blue-700 hover:text-blue-800 font-medium mt-2 inline-block" target="_blank" href="${esc(url)}">Open →</a>` : ""}
    </div>
  `;
}

async function fetchModel(goal, k, model) {
  const res = await fetch("/recommend", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ goal, k, model, top_n_candidates: 200, explain: false })
  });
  const body = await res.json();
  if (!res.ok) {
    return { error: body.detail || "failed" };
  }
  return body;
}

async function run() {
  const goal = document.getElementById("goal").value;
  const k = Number(document.getElementById("k").value || 5);

  const status = document.getElementById("status");
  status.textContent = "Comparing…";

  const [a, b, c] = await Promise.all([
    fetchModel(goal, k, "tfidf"),
    fetchModel(goal, k, "embedding"),
    fetchModel(goal, k, "hybrid"),
  ]);

  const tfidf = document.getElementById("tfidf");
  const embedding = document.getElementById("embedding");
  const hybrid = document.getElementById("hybrid");

  tfidf.innerHTML = a.error ? `<div class="text-sm text-red-300">${esc(a.error)}</div>` : a.recommendations.map(card).join("");
  embedding.innerHTML = b.error ? `<div class="text-sm text-red-300">${esc(b.error)}</div>` : b.recommendations.map(card).join("");
  hybrid.innerHTML = c.error ? `<div class="text-sm text-red-300">${esc(c.error)}</div>` : c.recommendations.map(card).join("");

  status.textContent = "";
}

document.getElementById("run").addEventListener("click", run);