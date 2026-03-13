function esc(s) {
  return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}

function pill(text, variant = "neutral") {
  const base = "inline-flex items-center px-2.5 py-1 rounded-full text-xs border";
  const styles = {
    neutral: "bg-slate-50 border-slate-200 text-slate-700",
    blue: "bg-blue-50 border-blue-200 text-blue-800",
    green: "bg-green-50 border-green-200 text-green-800"
  };
  return `<span class="${base} ${styles[variant] || styles.neutral}">${esc(text)}</span>`;
}

function scorePill(label, value, variant = "neutral") {
  return pill(`${label}: ${Number(value).toFixed(3)}`, variant);
}

function renderRec(rec) {
  const meta = rec.metadata || {};
  const ev = rec.evidence || {};

  const url = meta.url || "";
  const subject = meta.subject || "";
  const level = meta.level || "";
  const provider = meta.provider || "";
  const confidence = meta.confidence;

  const breakdown =
    meta.embedding_score !== undefined ||
    meta.tfidf_score !== undefined ||
    meta.meta_prior !== undefined ||
    confidence !== undefined
      ? `
      <div class="mt-4 pt-4 border-t border-slate-100">
        <div class="text-xs uppercase tracking-wide text-slate-400 font-medium">Model breakdown</div>
        <div class="mt-2 flex flex-wrap gap-2">
          ${meta.embedding_score !== undefined ? scorePill("emb", meta.embedding_score, "blue") : ""}
          ${meta.tfidf_score !== undefined ? scorePill("tfidf", meta.tfidf_score) : ""}
          ${meta.meta_prior !== undefined ? scorePill("prior", meta.meta_prior) : ""}
          ${meta.level_adjustment !== undefined ? scorePill("level", meta.level_adjustment, "green") : ""}
          ${confidence !== undefined && confidence !== null ? scorePill("conf", confidence, "green") : ""}
        </div>
      </div>
    `
      : "";

  return `
    <article class="bg-white border border-slate-200 rounded-[1.6rem] shadow-sm p-5 md:p-6 hover:shadow-md transition duration-200">
      <div class="flex flex-col md:flex-row md:items-start md:justify-between gap-5">
        <div class="flex-1">
          <div class="flex flex-wrap items-center gap-2">
            ${provider ? pill(provider) : ""}
            ${subject ? pill(subject, "blue") : ""}
            ${level ? pill(level) : ""}
          </div>

          <h3 class="mt-4 text-xl font-semibold tracking-tight text-slate-900">
            ${esc(rec.title)}
          </h3>

          ${rec.why ? `
            <p class="mt-3 text-slate-600 leading-7">
              ${esc(rec.why)}
            </p>
          ` : ""}
        </div>

        <div class="md:text-right min-w-[90px]">
          <div class="text-xs uppercase tracking-wide text-slate-400 font-medium">Score</div>
          <div class="mt-1 text-2xl font-semibold tracking-tight text-slate-900">
            ${Number(rec.score).toFixed(3)}
          </div>
        </div>
      </div>

      ${breakdown}

      <div class="mt-5 flex flex-wrap items-center gap-3">
        ${url ? `
          <a
            class="inline-flex items-center px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium transition"
            href="${esc(url)}"
            target="_blank"
            rel="noopener noreferrer"
          >
            Visit course
          </a>
        ` : ""}

        <details class="group">
          <summary class="list-none cursor-pointer inline-flex items-center px-4 py-2 rounded-xl border border-slate-200 hover:border-blue-300 hover:bg-blue-50 text-sm text-slate-700 font-medium transition">
            View evidence
          </summary>

          <div class="mt-3 rounded-2xl bg-slate-50 border border-slate-200 p-4 text-sm text-slate-600 space-y-2">
            ${ev.method ? `<div><span class="font-medium text-slate-800">Method:</span> ${esc(ev.method)}</div>` : ""}
            ${ev.model_name ? `<div><span class="font-medium text-slate-800">Model:</span> ${esc(ev.model_name)}</div>` : ""}
            ${ev.candidate_pool_size ? `<div><span class="font-medium text-slate-800">Candidate pool:</span> ${esc(ev.candidate_pool_size)}</div>` : ""}
            ${(ev.matched_terms && ev.matched_terms.length)
              ? `<div><span class="font-medium text-slate-800">Matched terms:</span> <div class="mt-2 flex flex-wrap gap-2">${ev.matched_terms.map(t => pill(t, "blue")).join("")}</div></div>`
              : `<div class="text-slate-500">No explicit matched terms were exposed for this result.</div>`
            }
          </div>
        </details>
      </div>
    </article>
  `;
}

async function run() {
  const goalEl = document.getElementById("goal");
  const modelEl = document.getElementById("model");
  const kEl = document.getElementById("k");
  const explainEl = document.getElementById("explain");
  const statusEl = document.getElementById("status");
  const resultsEl = document.getElementById("results");
  const emptyStateEl = document.getElementById("emptyState");

  const goal = goalEl?.value?.trim() || "";
  const model = modelEl?.value || "hybrid";
  const k = Number(kEl?.value || 5);
  const explain = Boolean(explainEl?.checked);

  if (!goal) {
    statusEl.textContent = "Please enter a learning goal before requesting recommendations.";
    statusEl.className = "mt-4 text-sm text-red-600";
    return;
  }

  statusEl.textContent = "Generating recommendations…";
  statusEl.className = "mt-4 text-sm text-slate-500";

  resultsEl.innerHTML = "";
  if (emptyStateEl) emptyStateEl.style.display = "none";

  const payload = {
    goal,
    k,
    model,
    top_n_candidates: 200,
    explain
  };

  try {
    const res = await fetch("/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const body = await res.json();

    if (!res.ok) {
      statusEl.textContent = `Error: ${body.detail || body.message || "Request failed."}`;
      statusEl.className = "mt-4 text-sm text-red-600";
      if (emptyStateEl) emptyStateEl.style.display = "block";
      return;
    }

    const recs = body.recommendations || [];

    if (!recs.length) {
      statusEl.textContent = "We couldn’t find a strong course match for that goal in the current dataset. Try a broader topic or rephrase your goal.";
      statusEl.className = "mt-4 text-sm text-amber-700";
      if (emptyStateEl) {
        emptyStateEl.style.display = "block";
        emptyStateEl.innerHTML = `
          <div class="bg-amber-50 border border-amber-200 rounded-2xl p-5 text-amber-900">
            <h3 class="text-lg font-semibold">No strong match found</h3>
            <p class="mt-2 text-sm leading-6">
              KimyGuide could not find a sufficiently relevant course for this goal in the current dataset.
              Try using a broader topic, simpler wording, or a closely related subject area.
            </p>
          </div>
        `;
      }
      return;
    }

    resultsEl.innerHTML = recs.map(renderRec).join("");
    statusEl.textContent = `Showing ${recs.length} recommendation${recs.length === 1 ? "" : "s"} using ${body.model_version || model}.`;
    statusEl.className = "mt-4 text-sm text-green-700";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Something went wrong while contacting the recommendation service.";
    statusEl.className = "mt-4 text-sm text-red-600";
    if (emptyStateEl) emptyStateEl.style.display = "block";
  }
}

document.getElementById("run")?.addEventListener("click", run);

// Allow Enter/Ctrl+Enter experience
document.getElementById("goal")?.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    run();
  }
});