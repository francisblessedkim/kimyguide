function esc(s) {
  return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

async function load() {
  const limit = Number(document.getElementById("limit").value || 50);
  const status = document.getElementById("status");
  const tbody = document.getElementById("tbody");

  status.textContent = "Loading…";
  tbody.innerHTML = "";

  const res = await fetch(`/dataset/sample?limit=${encodeURIComponent(limit)}`);
  const body = await res.json();

  if (!res.ok) {
    status.textContent = `Error: ${body.detail || "failed"}`;
    return;
  }

  const rows = body.rows || [];
  tbody.innerHTML = rows.map(r => `
    <tr class="border-t border-slate-800">
      <td class="py-2">${esc(r.title)}</td>
      <td class="py-2">${esc(r.subject)}</td>
      <td class="py-2">${esc(r.level)}</td>
      <td class="py-2"><a class="text-indigo-400 hover:text-indigo-300" href="${esc(r.url)}" target="_blank">Open →</a></td>
    </tr>
  `).join("");

  status.textContent = "";
}

document.getElementById("load").addEventListener("click", load);