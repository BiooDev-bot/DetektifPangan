/* RadarPangan — dashboard statis (baca data/*.json hasil pipeline). */
(function () {
  "use strict";

  const LEVELS = ["Rendah", "Waspada", "Tinggi", "Sedang lonjak"];
  const LEVEL_VAR = ["--low", "--watch", "--high", "--spike"];
  const state = { meta: null, risk: [], geo: null, net: {}, history: {}, commodity: null, province: null };
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const pct = (x, d = 0) => (x === null || x === undefined || !isFinite(x) ? "–" : (x * 100).toFixed(d) + "%");
  const signed = (x, d = 1) => (x === null || x === undefined || !isFinite(x) ? "–" : (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%");
  const fmtDate = (s) => new Date(s + "T00:00:00").toLocaleDateString("id-ID", { day: "numeric", month: "short", year: "numeric" });
  const rupiah = (x) => (isFinite(x) ? "Rp" + Math.round(x).toLocaleString("id-ID") : "–");

  async function getJSON(path) {
    const r = await fetch(path, { cache: "no-cache" });
    if (!r.ok) throw new Error(path + " -> HTTP " + r.status);
    return r.json();
  }

  function provName(pid) {
    const f = state.geo.features.find((g) => g.properties.province_id === pid);
    return f ? f.properties.province : "Provinsi " + pid;
  }

  function levelColor(level) {
    const i = LEVELS.indexOf(level);
    return css(LEVEL_VAR[i < 0 ? 0 : i]);
  }
  // warna untuk TEKS/badge: hijau muda "Rendah" terlalu pucat di atas latar terang
  function levelInk(level) {
    return level === "Rendah" ? css("--accent") : levelColor(level);
  }

  function baseLayout(extra) {
    return Object.assign({
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: css("--ink"), family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif" },
      dragmode: false,
      showlegend: false,
      geo: {
        visible: false, projection: { type: "mercator" },
        lonaxis: { range: [94.6, 141.2] }, lataxis: { range: [-11.2, 6.2] },
        domain: { x: [0, 1], y: [0, 1] }, bgcolor: "rgba(0,0,0,0)", showframe: false,
      },
    }, extra || {});
  }
  const plotCfg = { displayModeBar: false, responsive: true, scrollZoom: false };

  // ---------------------------------------------------------------- init
  async function init() {
    try {
      const [meta, risk, geo, net] = await Promise.all([
        getJSON("data/meta.json"), getJSON("data/risk_latest.json"),
        getJSON("data/provinces.geojson"), getJSON("data/network.json"),
      ]);
      Object.assign(state, { meta, risk, geo, net });
    } catch (e) {
      $("asof").textContent = "Gagal memuat data: " + e.message;
      return;
    }
    const m = state.meta;
    $("asof").textContent = `Data s.d. ${fmtDate(m.asof)} · risiko ${fmtDate(m.window.start)} – ${fmtDate(m.window.end)}`;
    buildSelect();
    buildLegend();
    buildAbout();
    const params = new URLSearchParams(location.search);
    setCommodity(params.get("k") || m.default_commodity);
  }

  function buildSelect() {
    const sel = $("commodity");
    const groups = {};
    state.meta.commodities.forEach((c) => { (groups[c.category] = groups[c.category] || []).push(c); });
    Object.keys(groups).forEach((cat) => {
      const og = document.createElement("optgroup");
      og.label = cat;
      groups[cat].forEach((c) => {
        const o = document.createElement("option");
        o.value = c.id; o.textContent = c.name;
        og.appendChild(o);
      });
      sel.appendChild(og);
    });
    sel.addEventListener("change", () => setCommodity(sel.value));
  }

  function buildLegend() {
    $("legend").innerHTML = LEVELS.map((l) => `<span><i style="background:${levelColor(l)}"></i>${l}</span>`).join("")
      + `<span><i style="background:${css("--nodata")}"></i>Tidak ada data</span>`;
  }

  function buildAbout() {
    const mt = state.meta.metrics || {};
    const items = [
      [mt.pr_auc_lgbm !== undefined ? mt.pr_auc_lgbm.toFixed(2) : "–", `PR-AUC model (baseline "harga minggu lalu": ${mt.pr_auc_naive !== undefined ? mt.pr_auc_naive.toFixed(2) : "–"})`],
      [pct(mt.event_detection_rate), "kejadian lonjakan yang didahului alarm (uji 2022–2026)"],
      [mt.median_lead_days !== undefined && mt.median_lead_days !== null ? mt.median_lead_days.toFixed(0) + " hari" : "–", "median jarak alarm pertama sebelum lonjakan"],
    ];
    $("metrics").innerHTML = items.map(([v, l]) => `<div class="metric"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
    $("sources").innerHTML = state.meta.sources;
  }

  // ---------------------------------------------------------------- commodity
  function setCommodity(cid) {
    const exists = state.meta.commodities.some((c) => c.id === cid);
    state.commodity = exists ? cid : state.meta.default_commodity;
    $("commodity").value = state.commodity;
    const c = state.meta.commodities.find((x) => x.id === state.commodity);
    $("threshold-hint").textContent = `Lonjakan = harga konsumen naik > ${pct(c.threshold, c.threshold < 0.1 ? 1 : 0)} dalam 14 hari. Alarm = peluang lonjakan dalam 14 hari ke depan ≥ ${pct(state.meta.alarm_threshold)}.`;
    const url = new URL(location.href); url.searchParams.set("k", state.commodity); history.replaceState(null, "", url);
    const rows = state.risk.filter((r) => r.c === state.commodity);
    renderSummary(rows);
    renderMap(rows);
    renderRanking(rows);
    renderNetwork();
    $("detail").hidden = true;
    const top = rows.slice().sort((a, b) => b.prob - a.prob)[0];
    if (top) showDetail(top.p, false);
  }

  function renderSummary(rows) {
    const count = (l) => rows.filter((r) => r.level === l).length;
    const cells = [["Sedang lonjak", count("Sedang lonjak")], ["Tinggi", count("Tinggi")], ["Waspada", count("Waspada")], ["Rendah", count("Rendah")]];
    $("summary").innerHTML = cells.map(([l, n]) =>
      `<div class="chip"><div class="n">${n}</div><div class="t"><span class="dot" style="background:${levelColor(l)}"></span>${l === "Rendah" ? "Risiko rendah" : l === "Tinggi" ? "Risiko tinggi" : l}</div></div>`).join("");
  }

  function renderMap(rows) {
    const byP = new Map(rows.map((r) => [r.p, r]));
    const all = state.geo.features.map((f) => f.properties.province_id);
    const nodata = all.filter((p) => !byP.has(p));
    const colors = LEVEL_VAR.map(css);
    const scale = [[0, colors[0]], [0.25, colors[0]], [0.25, colors[1]], [0.5, colors[1]], [0.5, colors[2]], [0.75, colors[2]], [0.75, colors[3]], [1, colors[3]]];
    const traces = [{
      type: "choropleth", geojson: state.geo, featureidkey: "properties.province_id",
      locations: nodata, z: nodata.map(() => 0), showscale: false,
      colorscale: [[0, css("--nodata")], [1, css("--nodata")]],
      marker: { line: { color: css("--card"), width: 0.6 } },
      hovertemplate: "%{text}<br>Tidak ada data<extra></extra>", text: nodata.map(provName),
    }, {
      type: "choropleth", geojson: state.geo, featureidkey: "properties.province_id",
      locations: rows.map((r) => r.p), z: rows.map((r) => LEVELS.indexOf(r.level)), zmin: 0, zmax: 3,
      colorscale: scale, showscale: false,
      marker: { line: { color: css("--card"), width: 0.6 } },
      text: rows.map((r) => `<b>${provName(r.p)}</b><br>${r.level} · peluang ${pct(r.prob)}<br>Perubahan 14 hari: ${signed(r.r14)}`),
      hovertemplate: "%{text}<extra></extra>",
    }];
    Plotly.react("map", traces, baseLayout(), plotCfg).then((gd) => {
      gd.removeAllListeners && gd.removeAllListeners("plotly_click");
      gd.on("plotly_click", (ev) => { const p = ev.points && ev.points[0]; if (p && byP.has(p.location)) showDetail(p.location, true); });
    });
  }

  function renderRanking(rows) {
    const top = rows.slice().sort((a, b) => b.prob - a.prob).slice(0, 10);
    $("ranking").innerHTML = top.map((r) => `
      <li data-p="${r.p}">
        <div class="row"><span class="name">${provName(r.p)}</span><span class="p" style="color:${levelInk(r.level)}">${pct(r.prob)}</span></div>
        <div class="why">${r.level}${r.reasons && r.reasons.length ? " · " + r.reasons[0] : ""}</div>
      </li>`).join("");
    $("ranking").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => showDetail(+li.dataset.p, true)));
  }

  // ---------------------------------------------------------------- detail
  async function showDetail(pid, scroll) {
    const r = state.risk.find((x) => x.c === state.commodity && x.p === pid);
    if (!r) return;
    state.province = pid;
    $("detail").hidden = false;
    $("detail-title").textContent = provName(pid);
    const badge = $("detail-badge");
    badge.textContent = r.level; badge.style.background = levelInk(r.level);
    $("detail-prob").innerHTML = r.level === "Sedang lonjak"
      ? `Harga <b>sedang melonjak</b>: ${signed(r.r14)} dalam 14 hari terakhir (${rupiah(r.price)}/kg).`
      : `Peluang lonjakan dalam 14 hari ke depan: <b>${pct(r.prob)}</b> · harga kini ${rupiah(r.price)}/kg (${signed(r.r14)} dalam 14 hari).`;
    $("detail-reasons").innerHTML = (r.reasons || []).map((t) => `<li>${t}</li>`).join("") || "<li>Tidak ada pendorong risiko yang menonjol.</li>";
    const lead = (state.net[state.commodity] || {}).leaders_of || {};
    const ls = lead[String(pid)] || [];
    $("detail-leaders").textContent = ls.length ? `Provinsi yang biasanya naik lebih dulu dari ${provName(pid)}: ${ls.map(provName).join(", ")}.` : "";
    if (scroll) $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
    try {
      if (!state.history[state.commodity]) state.history[state.commodity] = await getJSON(`data/history/${state.commodity}.json`);
      renderChart(state.history[state.commodity], pid);
    } catch (e) { $("detail-chart").textContent = "Riwayat harga tidak tersedia."; }
  }

  function renderChart(h, pid) {
    const key = String(pid);
    const traces = [];
    if (h.consumer && h.consumer[key]) traces.push({ x: h.dates, y: h.consumer[key], name: "Harga konsumen", type: "scatter", mode: "lines", line: { color: css("--accent"), width: 2.2 } });
    if (h.producer && h.producer[key]) traces.push({ x: h.dates, y: h.producer[key], name: "Harga produsen", type: "scatter", mode: "lines", line: { color: css("--muted"), width: 1.6, dash: "dot" } });
    if (h.national) traces.push({ x: h.dates, y: h.national, name: "Rata-rata nasional", type: "scatter", mode: "lines", line: { color: css("--watch"), width: 1.2 }, opacity: 0.8 });
    const ons = (h.onsets && h.onsets[key]) || [];
    if (ons.length && h.consumer && h.consumer[key]) {
      const idx = ons.map((d) => h.dates.indexOf(d)).filter((i) => i >= 0);
      traces.push({ x: idx.map((i) => h.dates[i]), y: idx.map((i) => h.consumer[key][i]), name: "Awal lonjakan", type: "scatter", mode: "markers", marker: { color: css("--high"), size: 10, symbol: "triangle-up" } });
    }
    const layout = {
      margin: { l: 56, r: 8, t: 8, b: 30 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: css("--ink"), size: 11 }, showlegend: true, legend: { orientation: "h", y: -0.18 },
      xaxis: { gridcolor: css("--line") }, yaxis: { gridcolor: css("--line"), tickprefix: "Rp", separatethousands: true },
      dragmode: false, hovermode: "x unified",
    };
    Plotly.react("detail-chart", traces, layout, plotCfg);
  }

  // ---------------------------------------------------------------- network
  function renderNetwork() {
    const n = state.net[state.commodity];
    const centroid = {};
    state.meta.centroids.forEach((c) => { centroid[c.p] = [c.lat, c.lon]; });
    const traces = [{
      type: "choropleth", geojson: state.geo, featureidkey: "properties.province_id",
      locations: state.geo.features.map((f) => f.properties.province_id), z: state.geo.features.map(() => 0),
      colorscale: [[0, css("--nodata")], [1, css("--nodata")]], showscale: false, hoverinfo: "skip",
      marker: { line: { color: css("--card"), width: 0.5 } },
    }];
    if (!n || !n.edges || !n.edges.length) {
      $("net-caption").textContent = "Belum ada pola rambatan yang signifikan untuk komoditas ini.";
      $("leaders").innerHTML = "";
      Plotly.react("netmap", traces, baseLayout(), plotCfg);
      return;
    }
    const maxS = Math.max(...n.edges.map((e) => e[2]));
    n.edges.forEach(([s, t, strength]) => {
      const a = centroid[s], b = centroid[t];
      if (!a || !b) return;
      traces.push({
        type: "scattergeo", mode: "lines", lat: [a[0], b[0]], lon: [a[1], b[1]], hoverinfo: "skip",
        line: { width: 1 + 3 * strength / maxS, color: css("--high") }, opacity: 0.35 + 0.5 * strength / maxS,
      });
    });
    const tgt = [...new Set(n.edges.map((e) => e[1]))];
    traces.push({
      type: "scattergeo", mode: "markers", lat: tgt.map((p) => centroid[p][0]), lon: tgt.map((p) => centroid[p][1]),
      marker: { size: 7, color: css("--card"), line: { color: css("--high"), width: 2 } },
      text: tgt.map(provName), hovertemplate: "%{text} (mengikuti)<extra></extra>",
    });
    const lead = n.leaders.filter((l) => l[1] > 0).slice(0, 8);
    const maxL = Math.max(1, ...lead.map((l) => l[1]));
    traces.push({
      type: "scattergeo", mode: "markers+text", lat: lead.map((l) => centroid[l[0]][0]), lon: lead.map((l) => centroid[l[0]][1]),
      marker: { size: lead.map((l) => 9 + 15 * l[1] / maxL), color: css("--high"), line: { color: css("--card"), width: 1 } },
      text: lead.map((l, i) => (i < 4 ? provName(l[0]) : "")), textposition: "top center", textfont: { size: 10, color: css("--ink") },
      customdata: lead.map((l) => provName(l[0])),
      hovertemplate: "%{customdata}: pemimpin<extra></extra>",
    });
    Plotly.react("netmap", traces, baseLayout(), plotCfg);
    $("net-caption").textContent = `Garis = kenaikan harga di satu provinsi secara statistik mendahului provinsi lain (uji Granger, dikontrol tren nasional). Titik besar = provinsi "pemimpin"; lingkaran kecil = provinsi yang mengikuti.`;
    $("leaders").innerHTML = lead.slice(0, 5).map((l) => `<li><div class="row"><span class="name">${provName(l[0])}</span><span class="p">memimpin ${l[2]} · mengikuti ${l[3]}</span></div></li>`).join("");
  }

  document.addEventListener("DOMContentLoaded", init);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.commodity && setCommodity(state.commodity));
})();
