function esc(s) {
  return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function pill(text, variant = "neutral") {
  const base = "inline-flex items-center px-2.5 py-1 rounded-full text-xs border";
  const styles = {
    neutral: "bg-slate-50 border-slate-200 text-slate-700",
    blue: "bg-blue-50 border-blue-200 text-blue-800",
    violet: "bg-violet-50 border-violet-200 text-violet-800",
    green: "bg-emerald-50 border-emerald-200 text-emerald-800"
  };
  return `<span class="${base} ${styles[variant] || styles.neutral}">${esc(text)}</span>`;
}

function emptyCard(message) {
  return `
    <div class="bg-white/85 border border-slate-200 rounded-[1.5rem] shadow-sm p-5 text-sm text-slate-500">
      ${esc(message)}
    </div>
  `;
}

function skeletonCard() {
  return `
    <div class="bg-white/85 border border-slate-200 rounded-[1.5rem] shadow-sm p-5 animate-pulse">
      <div class="h-4 w-24 bg-slate-200 rounded-full"></div>
      <div class="mt-4 h-5 w-4/5 bg-slate-200 rounded-lg"></div>
      <div class="mt-2 h-4 w-1/3 bg-slate-200 rounded-lg"></div>
      <div class="mt-5 h-9 w-24 bg-slate-200 rounded-xl"></div>
    </div>
  `;
}

function card(rec, accent = "blue") {
  const meta = rec.metadata || {};
  const url = meta.url || "";
  const subject = meta.subject || "";
  const level = meta.level || "";
  const provider = meta.provider || "";

  return `
    <article class="bg-white/90 border border-slate-200 rounded-[1.6rem] shadow-sm p-5 hover:shadow-lg hover:-translate-y-1 transition duration-300">
      <div class="flex items-start justify-between gap-4">
        <div class="flex-1">
          <div class="flex flex-wrap gap-2">
            ${provider ? pill(provider) : ""}
            ${subject ? pill(subject, accent) : ""}
            ${level ? pill(level, "green") : ""}
          </div>
          <h3 class="mt-4 text-lg font-semibold tracking-tight text-slate-900">${esc(rec.title)}</h3>
        </div>

        <div class="text-right shrink-0">
          <div class="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Score</div>
          <div class="mt-1 text-2xl font-semibold text-slate-900">${Number(rec.score).toFixed(3)}</div>
        </div>
      </div>

      <div class="mt-5 h-2 rounded-full bg-slate-100 overflow-hidden">
        <div class="h-full rounded-full ${accent === "violet" ? "bg-violet-500" : accent === "green" ? "bg-emerald-500" : "bg-blue-600"}" style="width:${Math.max(8, Math.min(100, Number(rec.score) * 100))}%"></div>
      </div>

      <div class="mt-5 flex flex-wrap gap-3">
        ${url ? `<a class="inline-flex items-center px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-sm font-medium transition" target="_blank" rel="noopener noreferrer" href="${esc(url)}">Visit course</a>` : ""}
      </div>
    </article>
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
    return { error: body.detail || "Request failed." };
  }
  return body;
}

async function run() {
  const goal = document.getElementById("goal")?.value?.trim() || "";
  const k = Number(document.getElementById("k")?.value || 5);
  const status = document.getElementById("status");
  const compareLoader = document.getElementById("compareLoader");

  const tfidf = document.getElementById("tfidf");
  const embedding = document.getElementById("embedding");
  const hybrid = document.getElementById("hybrid");

  if (!goal) {
    status.textContent = "Please enter a learning goal before comparing models.";
    status.className = "text-sm text-red-600";
    return;
  }

  status.textContent = "";
  compareLoader?.classList.remove("hidden");
  compareLoader?.classList.add("flex");

  tfidf.innerHTML = skeletonCard() + skeletonCard();
  embedding.innerHTML = skeletonCard() + skeletonCard();
  hybrid.innerHTML = skeletonCard() + skeletonCard();

  try {
    const [a, b, c] = await Promise.all([
      fetchModel(goal, k, "tfidf"),
      fetchModel(goal, k, "embedding"),
      fetchModel(goal, k, "hybrid"),
    ]);

    tfidf.innerHTML = a.error
      ? emptyCard(a.error)
      : (a.recommendations?.length ? a.recommendations.map((r) => card(r, "neutral")).join("") : emptyCard("No results returned for TF-IDF."));

    embedding.innerHTML = b.error
      ? emptyCard(b.error)
      : (b.recommendations?.length ? b.recommendations.map((r) => card(r, "violet")).join("") : emptyCard("No results returned for Embedding."));

    hybrid.innerHTML = c.error
      ? emptyCard(c.error)
      : (c.recommendations?.length ? c.recommendations.map((r) => card(r, "blue")).join("") : emptyCard("No results returned for Hybrid."));

    status.textContent = `Model comparison complete for "${goal}".`;
    status.className = "text-sm text-emerald-700";
  } catch (err) {
    console.error(err);
    status.textContent = "Something went wrong while comparing the models.";
    status.className = "text-sm text-red-600";
    tfidf.innerHTML = emptyCard("Unable to load TF-IDF results.");
    embedding.innerHTML = emptyCard("Unable to load Embedding results.");
    hybrid.innerHTML = emptyCard("Unable to load Hybrid results.");
  } finally {
    compareLoader?.classList.add("hidden");
    compareLoader?.classList.remove("flex");
  }
}

document.getElementById("run")?.addEventListener("click", run);

document.getElementById("goal")?.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    run();
  }
});