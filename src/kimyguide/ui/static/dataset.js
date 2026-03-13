function esc(s) {
  return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

let allRows = [];

function levelPill(level) {
  const val = (level || "").toLowerCase();
  let cls = "bg-slate-50 border-slate-200 text-slate-700";

  if (val.includes("intro")) cls = "bg-emerald-50 border-emerald-200 text-emerald-800";
  else if (val.includes("intermediate")) cls = "bg-amber-50 border-amber-200 text-amber-800";
  else if (val.includes("advanced")) cls = "bg-rose-50 border-rose-200 text-rose-800";

  return `<span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs border ${cls}">${esc(level || "—")}</span>`;
}

function skeletonRows(count = 6) {
  return Array.from({ length: count }).map(() => `
    <tr class="border-b border-slate-100 animate-pulse">
      <td class="py-4 px-4"><div class="h-4 w-56 bg-slate-200 rounded"></div></td>
      <td class="py-4 px-4"><div class="h-4 w-32 bg-slate-200 rounded"></div></td>
      <td class="py-4 px-4"><div class="h-6 w-24 bg-slate-200 rounded-full"></div></td>
      <td class="py-4 px-4"><div class="h-4 w-16 bg-slate-200 rounded"></div></td>
    </tr>
  `).join("");
}

function renderRows(rows) {
  const tbody = document.getElementById("tbody");
  tbody.innerHTML = rows.map((r) => `
    <tr class="border-b border-slate-100 hover:bg-blue-50/40 transition">
      <td class="py-4 px-4 font-medium text-slate-900">${esc(r.title)}</td>
      <td class="py-4 px-4 text-slate-700">${esc(r.subject)}</td>
      <td class="py-4 px-4">${levelPill(r.level)}</td>
      <td class="py-4 px-4">
        <a class="text-blue-700 hover:text-blue-800 font-medium" href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">
          Open →
        </a>
      </td>
    </tr>
  `).join("");
}

function applyFilter() {
  const q = (document.getElementById("search")?.value || "").trim().toLowerCase();
  const status = document.getElementById("status");

  const filtered = !q
    ? allRows
    : allRows.filter((r) =>
        `${r.title || ""} ${r.subject || ""} ${r.level || ""}`.toLowerCase().includes(q)
      );

  renderRows(filtered);

  if (status) {
    status.textContent = q
      ? `Showing ${filtered.length} filtered row${filtered.length === 1 ? "" : "s"}.`
      : `Showing ${filtered.length} loaded row${filtered.length === 1 ? "" : "s"}.`;
    status.className = "text-sm text-slate-600";
  }
}

async function load() {
  const limit = Number(document.getElementById("limit")?.value || 50);
  const status = document.getElementById("status");
  const tbody = document.getElementById("tbody");
  const datasetLoader = document.getElementById("datasetLoader");

  if (status) {
    status.textContent = "";
  }

  datasetLoader?.classList.remove("hidden");
  datasetLoader?.classList.add("flex");
  tbody.innerHTML = skeletonRows(6);

  try {
    const res = await fetch(`/dataset/sample?limit=${encodeURIComponent(limit)}`);
    const body = await res.json();

    if (!res.ok) {
      throw new Error(body.detail || "Failed to load dataset sample.");
    }

    allRows = body.rows || [];
    renderRows(allRows);

    if (status) {
      status.textContent = `Loaded ${allRows.length} sampled row${allRows.length === 1 ? "" : "s"}.`;
      status.className = "text-sm text-emerald-700";
    }
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `
      <tr>
        <td colspan="4" class="py-6 px-4 text-red-600">Unable to load the dataset sample.</td>
      </tr>
    `;
    if (status) {
      status.textContent = err.message || "Something went wrong.";
      status.className = "text-sm text-red-600";
    }
  } finally {
    datasetLoader?.classList.add("hidden");
    datasetLoader?.classList.remove("flex");
  }
}

document.getElementById("load")?.addEventListener("click", load);
document.getElementById("search")?.addEventListener("input", applyFilter);