function esc(s) {
  return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function pill(text, variant="neutral") {
  const base = "inline-flex items-center px-2.5 py-1 rounded-full text-xs border";
  const styles = {
    neutral: "bg-slate-50 border-slate-200 text-slate-700",
    blue: "bg-blue-50 border-blue-200 text-blue-800",
  };
  return `<span class="${base} ${styles[variant] || styles.neutral}">${esc(text)}</span>`;
}

function renderRec(rec) {
  const meta = rec.metadata || {};
  const ev = rec.evidence || {};
  const url = meta.url || "";
  const subject = meta.subject || "";
  const level = meta.level || "";
  const provider = meta.provider || "";
  const confidence = meta.confidence;

  const breakdown = (meta.embedding_score !== undefined || meta.tfidf_score !== undefined)
    ? `
      <div class="mt-3 text-xs text-slate-600">
        <div class="flex flex-wrap gap-2">
          ${meta.embedding_score !== undefined ? pill(`emb: ${Number(meta.embedding_score).toFixed(3)}`, "blue") : ""}
          ${meta.tfidf_score !== undefined ? pill(`tfidf: ${Number(meta.tfidf_score).toFixed(3)}`) : ""}
          ${meta.meta_prior !== undefined ? pill(`prior: ${Number(meta.meta_prior).toFixed(3)}`) : ""}
          ${confidence !== undefined && confidence !== null ? pill(`conf: ${Number(confidence).toFixed(3)}`, "blue") : ""}
        </div>
      </div>
    ` : "";

  return `
  <div class="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
    <div class="flex items-start justify-between gap-4">
      <div>
        <div class="text-lg font-semibold tracking-tight">${esc(rec.title)}</div>
        <div class="mt-2 flex flex-wrap gap-2">
          ${provider ? pill(provider) : ""}
          ${subject ? pill(subject, "blue") : ""}
          ${level ? pill(level) : ""}
        </div>
      </div>

      <div class="text-right min-w-[72px]">
        <div class="text-xs text-slate-500">score</div>
        <div class="text-2xl font-semibold tracking-tight">${Number(rec.score).toFixed(3)}</div>
      </div>
    </div>

    ${rec.why ? `<p class="mt-4 text-slate-700 leading-relaxed">${esc(rec.why)}</p>` : ""}

    ${breakdown}

    <details class="mt-4">
      <summary class="cursor-pointer text-sm text-blue-700 hover:text-blue-800 select-none">Evidence</summary>
      <div class="mt-3 text-sm text-slate-600 space-y-2">
        ${ev.method ? `<div>Method: <span class="text-slate-900 font-medium">${esc(ev.method)}</span></div>` : ""}
        ${ev.model_name ? `<div>Model: <span class="text-slate-900 font-medium">${esc(ev.model_name)}</span></div>` : ""}
        ${ev.candidate_pool_size ? `<div>Candidate pool: <span class="text-slate-900 font-medium">${esc(ev.candidate_pool_size)}</span></div>` : ""}
        ${(ev.matched_terms && ev.matched_terms.length) ? `<div>Matched terms: ${ev.matched_terms.map(t => pill(t,"blue")).join(" ")}</div>` : ""}
        ${url ? `<div><a class="text-blue-700 hover:text-blue-800 font-medium" href="${esc(url)}" target="_blank">Open course →</a></div>` : ""}
      </div>
    </details>
  </div>`;
}

async function run() {
  const goal = document.getElementById("goal").value;
  const model = document.getElementById("model").value;
  const k = Number(document.getElementById("k").value || 5);
  const top_n_candidates = Number(document.getElementById("cand").value || 200);
  const explain = document.getElementById("explain").checked;

  const status = document.getElementById("status");
  const results = document.getElementById("results");
  const mv = document.getElementById("mv");

  results.innerHTML = "";
  mv.textContent = "";
  status.textContent = "Running…";

  const payload = { goal, k, model, top_n_candidates, explain };

  const res = await fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await res.json();

  if (!res.ok) {
    status.textContent = `Error: ${body.detail || body.message || "Request failed"}`;
    return;
  }

  status.textContent = "";
  mv.textContent = body.model_version || "";

  const recs = body.recommendations || [];
  results.innerHTML = recs.map(renderRec).join("");
}

document.getElementById("run").addEventListener("click", run);