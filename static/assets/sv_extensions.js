/* SafeVision Edge UI Extensions — Segmented Engine Control, Accurate Zone Safety Intelligence, Truthful NPU Telemetry & Reports + Zone Modal */

const ZONE_PALETTE = {
  road: { fill: "#3b6ea8", stroke: "#7fb0e8" },
  crossing: { fill: "#f0a202", stroke: "#f7b84b" },
  footpath: { fill: "#3e9d6e", stroke: "#6fd0a0" },
};

const CLASSES_7 = ["person", "bicycle", "motorcycle", "auto_rickshaw", "car", "bus", "truck"];
const HEAT_LABELS = ["Top-Left", "Top-Center", "Top-Right", "Mid-Left", "Mid-Center", "Mid-Right", "Low-Left", "Low-Center", "Low-Right"];
const PULSE_COLORS = { Low: "var(--lo)", Medium: "var(--me)", High: "var(--hi)" };

function getHost() {
  return window.__SV_HOST;
}

function h(type, props, ...children) {
  return getHost().React.createElement(type, props, ...children);
}

function inferZoneTypeJS(z) {
  const t = String((z && z.type) || "").trim().toLowerCase();
  if (t === "road" || t === "crossing" || t === "footpath") return t;
  const n = String((z && z.name) || "").trim().toLowerCase();
  if (n.includes("cross")) return "crossing";
  if (n.includes("foot") || n.includes("sidewalk") || n.includes("walk")) return "footpath";
  return "road";
}

function zonePaletteFor(z) {
  return ZONE_PALETTE[inferZoneTypeJS(z)] || { fill: "#7f56d9", stroke: "#d6bbfb" };
}

function cloneZones(zones) {
  return JSON.parse(JSON.stringify(zones || []));
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(Number(v).toFixed(5))));
}

function polyBounds(pts) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}

function pointInPoly(pt, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/* ============================================================================
 * 1. CLEAN 3-OPTION SEGMENTED ENGINE CONTROL (CPU | NPU | Compare Both)
 * ========================================================================== */
export function EngineSelectorBar({ engineStatus, selectedMode, onSelectMode }) {
  const { React, apiFetch } = getHost();
  const [busy, setBusy] = React.useState(false);
  const [errMsg, setErrMsg] = React.useState(null);

  const cpu = (engineStatus && engineStatus.cpu) || {};
  const npu = (engineStatus && engineStatus.npu) || {};
  const mode = selectedMode || (engineStatus && engineStatus.selected_mode) || "cpu";

  const cpuReady = cpu.available !== false;
  const npuReady = !!npu.available;
  const compareReady = !!(cpuReady && npuReady);

  const handleSelect = async (nextMode) => {
    setErrMsg(null);
    setBusy(true);
    try {
      const updated = await apiFetch("/engines/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: nextMode }),
      });
      if (onSelectMode) onSelectMode(updated.selected_mode || nextMode, updated);
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Failed to switch engine mode");
    } finally {
      setBusy(false);
    }
  };

  const npuSub = npuReady
    ? npu.model_name || "Hailo-8 Ready"
    : npu.status_code === "NPU_MODEL_NOT_READY"
    ? "Missing .hef model"
    : "Hardware unavailable";

  const options = [
    {
      id: "cpu",
      label: "CPU",
      sub: cpuReady ? "ExecuTorch · Ready" : "Unavailable",
      disabled: busy || !cpuReady,
    },
    {
      id: "npu",
      label: "NPU",
      sub: npuSub,
      disabled: busy,
    },
    {
      id: "compare",
      label: "Compare Both",
      sub: compareReady ? "CPU vs Hailo-8" : "Requires NPU",
      disabled: busy || !compareReady,
    },
  ];

  return h(
    "div",
    { className: "sv-seg-wrap" },
    h(
      "div",
      { className: "sv-seg-header" },
      h("span", { className: "sv-seg-title" }, "Inference Engine Mode"),
      h(
        "span",
        { className: `sv-status-badge ${mode === "cpu" ? (cpuReady ? "ready" : "err") : npuReady ? "ready" : "warn"}` },
        mode === "cpu"
          ? `CPU: ${cpu.model_name || "custom_model.pte"}`
          : mode === "npu"
          ? npuReady
            ? `NPU: ${npu.model_name || "custom_model.hef"}`
            : `NPU: ${npu.status_code || "Unavailable"}`
          : compareReady
          ? "Compare: CPU + NPU"
          : "Compare unavailable"
      )
    ),
    h(
      "div",
      { className: "sv-seg-control", role: "radiogroup", "aria-label": "Inference Engine Mode" },
      options.map((opt) =>
        h(
          "button",
          {
            key: opt.id,
            type: "button",
            role: "radio",
            "aria-checked": mode === opt.id,
            disabled: opt.disabled,
            className: `sv-seg-btn ${mode === opt.id ? "active" : ""}`,
            onClick: () => handleSelect(opt.id),
          },
          h("span", { className: "sv-seg-label" }, opt.label),
          h("span", { className: "sv-seg-sub" }, opt.sub)
        )
      )
    ),
    h(
      "div",
      { className: "sv-seg-footnote" },
      h("span", null, `Active CPU: ${cpu.runtime || "ExecuTorch"} + ${cpu.backend || "XNNPACK"} (${cpu.model_name || "custom_model.pte"})`),
      !npuReady ? h("span", null, `NPU unavailable: ${npu.message || "Requires Hailo-8 AI HAT+ & custom_model.hef"}`) : null
    ),
    errMsg ? h("div", { className: "alert", style: { marginTop: 8, padding: "8px 12px", fontSize: 13 } }, errMsg) : null
  );
}

/* ============================================================================
 * 2. SAFETY INTELLIGENCE PAGE (Explicit Occupancy, Classes, Calibration, Temporal)
 * ========================================================================== */
export function SafetyIntelligencePage({ results, selectedFrame, heatmap, safetyConfig, onSelectFrame, onNavigate }) {
  const { React, mediaUrl } = getHost();
  const [heatTab, setHeatTab] = React.useState("grid");

  if (!results || !selectedFrame) {
    return h(
      "div",
      null,
      h("h1", null, "Safety intelligence"),
      h("p", { className: "sub" }, "Zone occupancy, VRU share, co-presence alerts, and Safety Pulse."),
      h(
        "div",
        { className: "card empty" },
        h("h2", null, "No safety data"),
        h("p", null, "Run an inspection first to compute Safety Pulse scores and pedestrian heatmaps."),
        h("button", { type: "button", className: "btn", onClick: () => onNavigate("inspect") }, "Start an inspection")
      )
    );
  }

  const vruPct = Math.round((selectedFrame.vru_share || 0) * 100);
  const z = selectedFrame.zones || {};
  const roadZ = z.road || { person: selectedFrame.persons_road || 0, vehicles: selectedFrame.vehicles_road || 0 };
  const crossZ = z.crossing || { person: selectedFrame.persons_crossing || 0, vehicles: selectedFrame.vehicles_crossing || 0 };
  const footZ = z.footpath || { person: selectedFrame.persons_footpath || 0, vehicles: selectedFrame.vehicles_footpath || 0 };
  const unZ = z.unassigned || { total: 0, person: 0, vehicles: 0, by_class: {} };
  const byZoneNamed = z.by_zone || {};
  const byCatClasses = z.by_category_classes || {};
  const temporal = results.temporal_occupancy || null;
  const calDens = selectedFrame.calibrated_densities || { calibrated: false, physical_density_available: false, per_zone: {} };

  const heatCells = heatmap && Array.isArray(heatmap.cells) && heatmap.cells.length === 9 ? heatmap.cells : null;
  const heatMax = heatCells ? Math.max(1, ...heatCells) : 1;
  const heatImgUrl = heatmap && heatmap.image_url ? mediaUrl(heatmap.image_url) : null;

  // Dynamic rule text synchronized with backend safety_config.json thresholds
  const densBins = (safetyConfig && safetyConfig.density_thresholds) || { low_max: 4, moderate_max: 9 };
  const weights = (safetyConfig && safetyConfig.weights) || { density_moderate: 1, density_high: 2, co_presence: 2, high_vru_share: 1 };
  const bands = (safetyConfig && safetyConfig.safety_pulse_bands) || { low_max: 1, medium_max: 3 };
  const vruThreshPct = safetyConfig ? Math.round((safetyConfig.vru_share_threshold || 0.5) * 100) : 50;
  const vehClasses = (safetyConfig && Array.isArray(safetyConfig.vehicle_classes) ? safetyConfig.vehicle_classes : CLASSES_7.slice(1)).join(", ");

  const renderClassCells = (clsMap) =>
    CLASSES_7.map((cls) => h("td", { key: cls }, Number((clsMap && clsMap[cls]) || 0)));

  return h(
    "div",
    null,
    h("h1", null, "Safety intelligence"),
    h(
      "p",
      { className: "sub" },
      `Per-frame backend indicators for frame ${selectedFrame.frame_id} (${selectedFrame.source_label}). Scene: ${results.scene_id || "default"}.`
    ),

    // Frame ribbon selector
    h(
      "div",
      { className: "card" },
      h("h2", null, "Frames"),
      h(
        "div",
        { className: "ribbon", role: "group", "aria-label": "Safety Pulse per frame" },
        results.frames.map((fr) => {
          const sel = fr.frame_id === selectedFrame.frame_id;
          const lbl = (fr.safety_pulse && fr.safety_pulse.label) || "Low";
          const sc = (fr.safety_pulse && fr.safety_pulse.score) || 0;
          const height = 28 + Math.min(5, Math.max(0, sc)) * 7;
          return h("button", {
            key: fr.frame_id,
            type: "button",
            className: sel ? "sel" : undefined,
            style: { background: PULSE_COLORS[lbl] || "var(--lo)", height: `${height}px` },
            title: `Frame ${fr.frame_id} (${fr.source_label}): ${sc}/5 (${lbl})`,
            onClick: () => onSelectFrame(fr.frame_id),
          });
        })
      )
    ),

    // Top 4 KPIs with clear explicit labels (replacing "0 / 2")
    h(
      "div",
      { className: "grid g4", style: { marginTop: 16 } },
      h(
        "div",
        { className: "card kpi" },
        h("div", { className: "lbl" }, "Vehicle count & occupancy level"),
        h("div", { className: "val" }, selectedFrame.vehicles),
        h("div", { className: "sv-kpi-lines" }, h("span", null, `Relative level: `, h("b", null, selectedFrame.density)))
      ),
      h(
        "div",
        { className: "card kpi" },
        h("div", { className: "lbl" }, "Vulnerable road user share"),
        h("div", { className: "val" }, `${vruPct}%`),
        h("div", { className: "sv-kpi-lines" }, h("span", null, `${selectedFrame.vru_count} of ${selectedFrame.total} detections`))
      ),
      h(
        "div",
        { className: "card kpi" },
        h("div", { className: "lbl" }, "Crossing occupancy"),
        h(
          "div",
          { className: "sv-kpi-lines", style: { marginTop: 8 } },
          h("span", null, "Pedestrians: ", h("b", null, crossZ.person)),
          h("span", null, "Vehicles: ", h("b", null, crossZ.vehicles)),
          h("span", null, "Co-presence: ", h("b", null, selectedFrame.co_presence ? "Yes" : "No"))
        )
      ),
      h(
        "div",
        { className: "card kpi" },
        h("div", { className: "lbl" }, "Safety Pulse score"),
        h(
          "div",
          { className: "val", style: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" } },
          `${selectedFrame.safety_pulse.score}/5`,
          h("span", { className: `pill ${selectedFrame.safety_pulse.label}` }, selectedFrame.safety_pulse.label)
        )
      )
    ),

    // Backend Safety Pulse explanation banner
    h(
      "div",
      { className: selectedFrame.co_presence ? "alert" : "alert ok", style: { margin: "16px 0" } },
      h("strong", null, selectedFrame.co_presence ? "Co-presence detected in crossing zone" : "No pedestrian-vehicle co-presence in crossing zone"),
      h("p", { style: { margin: "4px 0 0" } }, selectedFrame.safety_pulse.explanation)
    ),

    // Main 2-column row: Heatmap + Zone occupancy & rules
    h(
      "div",
      { className: "grid g2" },
      h(
        "div",
        { className: "card" },
        h("h2", null, "Pedestrian heatmap"),
        heatmap && heatmap.available
          ? h(
              "div",
              null,
              heatImgUrl && heatCells
                ? h(
                    "div",
                    { className: "ctl", style: { marginTop: 0, marginBottom: 12 } },
                    h("button", { type: "button", className: heatTab === "grid" ? "btn" : "btn ghost", onClick: () => setHeatTab("grid") }, "3x3 Density Grid"),
                    h("button", { type: "button", className: heatTab === "image" ? "btn" : "btn ghost", onClick: () => setHeatTab("image") }, "Rendered Heatmap Image")
                  )
                : null,
              heatTab === "grid" && heatCells
                ? h(
                    "div",
                    null,
                    h(
                      "div",
                      { className: "heat" },
                      heatCells.map((cnt, idx) =>
                        h(
                          "div",
                          {
                            key: idx,
                            style: {
                              background: `rgba(127, 86, 217, ${(0.22 + (cnt / heatMax) * 0.72).toFixed(2)})`,
                            },
                          },
                          cnt,
                          h("small", null, HEAT_LABELS[idx])
                        )
                      )
                    ),
                    h("p", { className: "note", style: { marginTop: 8, marginBottom: 0 } }, `Bottom-centre points of detected people accumulated across ${results.total_frames} processed frames (${heatmap.total_pedestrian_positions || 0} total positions).`)
                  )
                : heatImgUrl
                ? h("img", { src: heatImgUrl, alt: "Pedestrian spatial heatmap", className: "frame" })
                : h("p", { className: "note" }, "Heatmap data unavailable.")
            )
          : h("p", { className: "note", style: { margin: 0 } }, "Pedestrian heatmap was not generated for this inspection.")
      ),

      h(
        "div",
        { className: "card" },
        h("h2", null, "Zone occupancy & indicator rules"),
        h(
          "div",
          { className: "sv-occ-list" },
          h(
            "div",
            { className: "sv-occ-row" },
            h("span", { className: "sv-occ-title" }, "Road occupancy"),
            h(
              "div",
              { className: "sv-occ-counts" },
              h("span", null, "Pedestrians: ", h("strong", null, roadZ.person)),
              h("span", null, "Vehicles: ", h("strong", null, roadZ.vehicles))
            )
          ),
          h(
            "div",
            { className: "sv-occ-row" },
            h("span", { className: "sv-occ-title" }, "Crossing occupancy"),
            h(
              "div",
              { className: "sv-occ-counts" },
              h("span", null, "Pedestrians: ", h("strong", null, crossZ.person)),
              h("span", null, "Vehicles: ", h("strong", null, crossZ.vehicles)),
              h("span", null, "Co-presence: ", h("strong", null, selectedFrame.co_presence ? "Yes" : "No"))
            )
          ),
          h(
            "div",
            { className: "sv-occ-row" },
            h("span", { className: "sv-occ-title" }, "Footpath occupancy (combined)"),
            h(
              "div",
              { className: "sv-occ-counts" },
              h("span", null, "Pedestrians: ", h("strong", null, footZ.person)),
              h("span", null, "Vehicles: ", h("strong", null, footZ.vehicles))
            )
          ),
          unZ.total > 0
            ? h(
                "div",
                { className: "sv-occ-row" },
                h("span", { className: "sv-occ-title" }, "Unassigned (outside all polygons)"),
                h(
                  "div",
                  { className: "sv-occ-counts" },
                  h("span", null, "Pedestrians: ", h("strong", null, unZ.person || 0)),
                  h("span", null, "Vehicles: ", h("strong", null, unZ.vehicles || 0)),
                  h("span", null, "Total: ", h("strong", null, unZ.total))
                )
              )
            : null
        ),

        // Calibrated metric density ONLY when homography calibration exists
        calDens.calibrated && calDens.physical_density_available
          ? h(
              "div",
              { className: "alert ok", style: { marginTop: 12, padding: "8px 12px", fontSize: 12.5 } },
              h("strong", null, "Calibrated Ground-Plane Density (Homography): "),
              Object.entries(calDens.per_zone).map(([zn, info]) =>
                h("div", { key: zn }, `${zn}: ${info.pedestrians_per_m2} ped/m² · ${info.vehicles_per_m2} veh/m² (area ${info.area_m2} m²)`)
              )
            )
          : h(
              "p",
              { className: "note", style: { marginTop: 10, marginBottom: 0 } },
              "Occupancy values show raw object counts inside calibrated polygons (physical m² density requires 4-point ground calibration in Reports and Settings)."
            ),

        // Score breakdown lines
        h(
          "dl",
          { style: { marginTop: 12 } },
          h("dt", null, "Vehicle occupancy contribution"),
          h("dd", null, `+${selectedFrame.density_score}`),
          h("dt", null, "Crossing co-presence contribution"),
          h("dd", null, `+${selectedFrame.co_presence_score}`),
          h("dt", null, "VRU share contribution"),
          h("dd", null, `+${selectedFrame.vru_score}`)
        ),

        // Collapsible Per-Class & Per-Zone Breakdown
        h(
          "details",
          { className: "sv-collapse" },
          h("summary", null, "Per-class & per-zone occupancy breakdown (7 classes)"),
          h(
            "div",
            { className: "scroll", style: { marginTop: 8 } },
            h(
              "table",
              null,
              h(
                "thead",
                null,
                h(
                  "tr",
                  null,
                  h("th", null, "Zone / Category"),
                  CLASSES_7.map((cls) => h("th", { key: cls }, cls.replace("_", " "))),
                  h("th", null, "Total")
                )
              ),
              h(
                "tbody",
                null,
                Object.keys(byZoneNamed).length > 0
                  ? Object.entries(byZoneNamed).map(([zname, zmap]) =>
                      h("tr", { key: zname }, h("td", null, h("strong", null, zname)), renderClassCells(zmap), h("td", null, h("strong", null, zmap.total || 0)))
                    )
                  : ["road", "crossing", "footpath"].map((cat) =>
                      h(
                        "tr",
                        { key: cat },
                        h("td", null, h("strong", null, cat)),
                        renderClassCells(byCatClasses[cat]),
                        h("td", null, h("strong", null, (byCatClasses[cat] && byCatClasses[cat].total) || 0))
                      )
                    ),
                h(
                  "tr",
                  null,
                  h("td", null, h("em", null, "Unassigned")),
                  renderClassCells(unZ.by_class),
                  h("td", null, unZ.total || 0)
                )
              )
            )
          )
        ),

        // Synchronized Rule reference + Disclaimer
        h(
          "div",
          { className: "note", style: { marginTop: 12 } },
          h("strong", null, "Rule reference: "),
          `Vehicle score: 0–${densBins.low_max} = +0, ${densBins.low_max + 1}–${densBins.moderate_max} = +${weights.density_moderate}, >=${densBins.moderate_max + 1} = +${weights.density_high}; Crossing co-presence = +${weights.co_presence}; VRU share >= ${vruThreshPct}% = +${weights.high_vru_share}. Bands: Low (0–${bands.low_max}), Medium (${bands.low_max + 1}–${bands.medium_max}), High (> ${bands.medium_max}). Vehicles include ${vehClasses}.`
        ),
        h(
          "p",
          { className: "note", style: { marginTop: 8, marginBottom: 0 } },
          results.indicator_disclaimer ||
            "Safety Pulse is an explainable, configurable rule-based operational indicator and is not a validated accident-prediction or statistical crash-risk model."
        )
      )
    ),

    // Temporal Occupancy Summary across multi-frame / video inspections
    temporal && temporal.frames_analyzed > 1
      ? h(
          "div",
          { className: "card", style: { marginTop: 16 } },
          h("h2", null, `Temporal occupancy summary (${temporal.frames_analyzed} frames)`),
          h(
            "div",
            { className: "sv-temporal-grid" },
            h("div", { className: "sv-mini-stat" }, h("small", null, "Average total / frame"), h("strong", null, temporal.avg_occupancy.total)),
            h("div", { className: "sv-mini-stat" }, h("small", null, "Peak single-frame occupancy"), h("strong", null, temporal.peak_occupancy.total)),
            h("div", { className: "sv-mini-stat" }, h("small", null, "Frames with pedestrians"), h("strong", null, `${temporal.pedestrian_presence_pct}% (${temporal.pedestrian_presence_frames}/${temporal.frames_analyzed})`)),
            h("div", { className: "sv-mini-stat" }, h("small", null, "Crossing co-presence rate"), h("strong", null, `${temporal.co_presence_pct}% (${temporal.co_presence_frames}/${temporal.frames_analyzed})`))
          ),
          h("p", { className: "note", style: { marginTop: 10, marginBottom: 0 } }, temporal.note)
        )
      : null
  );
}

/* ============================================================================
 * 3. TRUTHFUL NPU & BENCHMARK PAGE (Hide Unavailable Telemetry, No Mocked NPU)
 * ========================================================================== */
export function NpuBenchmarkPage({ engineStatus, onRefreshEngineStatus, inspections }) {
  const { React, apiFetch } = getHost();
  const [hw, setHw] = React.useState(null);
  const [latestCmp, setLatestCmp] = React.useState(null);
  const [mode, setMode] = React.useState("cpu");
  const [inspId, setInspId] = React.useState("");
  const [warmup, setWarmup] = React.useState(3);
  const [measured, setMeasured] = React.useState(10);
  const [confThres, setConfThres] = React.useState(0.25);
  const [job, setJob] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [errMsg, setErrMsg] = React.useState(null);

  const loadAll = React.useCallback(async () => {
    try {
      const [hRes, rRes] = await Promise.all([apiFetch("/hardware/status").catch(() => null), apiFetch("/benchmark/results").catch(() => null)]);
      if (hRes) setHw(hRes);
      if (rRes) setLatestCmp(rRes);
    } catch {}
  }, [apiFetch]);

  React.useEffect(() => {
    loadAll();
    const id = window.setInterval(loadAll, 4000);
    return () => window.clearInterval(id);
  }, [loadAll]);

  const cpu = (engineStatus && engineStatus.cpu) || (hw && hw.engines && hw.engines.cpu) || {};
  const npu = (engineStatus && engineStatus.npu) || (hw && hw.npu) || {};
  const cpuReady = cpu.available !== false;
  const npuReady = !!npu.available;
  const bothReady = cpuReady && npuReady;

  const runBenchmark = async () => {
    setErrMsg(null);
    setBusy(true);
    try {
      const started = await apiFetch("/benchmark/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          warmup_runs: Number(warmup),
          measured_runs: Number(measured),
          conf_thres: Number(confThres),
          iou_thres: 0.45,
          inspection_id: inspId || null,
        }),
      });
      setJob(started);
      const jid = started.job_id;
      for (let i = 0; i < 180; i++) {
        await new Promise((r) => setTimeout(r, 700));
        const st = await apiFetch(`/benchmark/compare/${encodeURIComponent(jid)}`);
        setJob(st);
        if (st.status === "completed") {
          if (st.report) setLatestCmp({ available: true, ...st.report, csv_available: true, csv_url: "/api/v1/benchmark/results/csv" });
          break;
        }
        if (st.status === "failed") {
          setErrMsg(st.error || "Benchmark failed");
          break;
        }
      }
    } catch (e) {
      setErrMsg(e instanceof Error ? e.message : "Failed to run benchmark");
    } finally {
      setBusy(false);
    }
  };

  const completedInsps = (inspections || []).filter((it) => it.status === "completed");
  const rep = latestCmp && latestCmp.available !== false && latestCmp.cpu ? latestCmp : null;

  // Show only real available hardware telemetry cards (hide unavailable temp/throttling/power tiles)
  const hwTiles = [];
  if (hw) {
    hwTiles.push({ label: "CPU Utilization", val: `${hw.system_cpu_percent}%`, sub: `Process: ${hw.process_cpu_percent}%` });
    hwTiles.push({
      label: "RAM Usage",
      val: `${hw.process_rss_mb} MB`,
      sub: `System: ${hw.system_ram_used_mb} / ${hw.system_ram_total_mb} MB`,
    });
    if (hw.cpu_temperature_c != null) {
      hwTiles.push({ label: "CPU Temperature", val: `${hw.cpu_temperature_c} °C`, sub: hw.cpu_freq_mhz ? `${hw.cpu_freq_mhz} MHz` : null });
    }
    if (hw.throttling && hw.throttling.supported) {
      hwTiles.push({ label: "Thermal Throttling", val: hw.throttling.status, sub: hw.throttling.raw_hex || null });
    }
    if (npuReady && npu.temperature_c != null) {
      hwTiles.push({ label: "Hailo-8 Temperature", val: `${npu.temperature_c} °C`, sub: npu.device_architecture || "HAILO8" });
    }
    if (hw.power && hw.power.available && hw.power.avg_watts != null) {
      hwTiles.push({ label: "Measured Power", val: `${hw.power.avg_watts} W`, sub: hw.power.source || "External meter" });
    }
  }

  return h(
    "div",
    null,
    h(
      "div",
      { style: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap", marginBottom: 16 } },
      h(
        "div",
        null,
        h("h1", { style: { marginBottom: 4 } }, "NPU & Benchmark"),
        h("p", { className: "sub", style: { marginBottom: 0 } }, "Truthful CPU / Hailo-8 readiness, measured host telemetry, and controlled performance benchmarking.")
      ),
      h(
        "button",
        {
          type: "button",
          className: "btn ghost",
          onClick: () => {
            if (onRefreshEngineStatus) onRefreshEngineStatus();
            loadAll();
          },
        },
        "Refresh Status"
      )
    ),

    // CPU & NPU Status
    h(
      "div",
      { className: "grid g2" },
      h(
        "div",
        { className: "card" },
        h(
          "div",
          { className: "ch" },
          h("h2", null, "CPU Engine"),
          h("span", { className: `sv-status-badge ${cpuReady ? "ready" : "err"}` }, cpuReady ? "READY" : "UNAVAILABLE")
        ),
        h(
          "dl",
          null,
          h("dt", null, "Selected Model"),
          h("dd", null, `${cpu.model_name || "custom_model.pte"} (${cpu.quantization || "INT8 PT2E"})`),
          h("dt", null, "Runtime"),
          h("dd", null, `${cpu.runtime || "ExecuTorch"} + ${cpu.backend || "XNNPACK"}`),
          h("dt", null, "Model Loaded"),
          h("dd", null, cpu.loaded ? "Yes" : "No")
        )
      ),
      h(
        "div",
        { className: "card" },
        h(
          "div",
          { className: "ch" },
          h("h2", null, "Hailo-8 NPU Engine"),
          h("span", { className: `sv-status-badge ${npuReady ? "ready" : npu.status_code === "NPU_MODEL_NOT_READY" ? "warn" : "err"}` }, npu.status_code || "UNAVAILABLE")
        ),
        h(
          "dl",
          null,
          h("dt", null, "Hailo-8 Hardware"),
          h("dd", null, npu.device_connected ? (npu.device_nodes || ["Detected"]).join(", ") : "Not detected"),
          h("dt", null, "HailoRT"),
          h("dd", null, npu.hailort_available ? npu.hailort_version || "Installed" : "Not installed"),
          h("dt", null, "Compiled .hef"),
          h("dd", null, npu.model_exists ? `${npu.model_name} (${npu.model_size_mb} MB)` : "Missing")
        ),
        !npuReady && npu.message ? h("p", { className: "note", style: { marginTop: 10, marginBottom: 0 } }, npu.message) : null
      )
    ),

    // Live Hardware Telemetry (only supported & measured tiles shown)
    hwTiles.length > 0
      ? h(
          "div",
          { className: "card", style: { marginTop: 16 } },
          h("h2", null, "Measured Hardware Telemetry"),
          h(
            "div",
            { className: "sv-hw-grid" },
            hwTiles.map((t, i) =>
              h(
                "div",
                { className: "sv-hw-tile", key: i },
                h("div", { className: "label" }, t.label),
                h("div", { className: "val" }, t.val),
                t.sub ? h("p", { className: "sub" }, t.sub) : null
              )
            )
          )
        )
      : null,

    // Controlled Benchmark Runner
    h(
      "div",
      { className: "card", style: { marginTop: 16 } },
      h("h2", null, "Run Controlled Benchmark"),
      h("p", { className: "note", style: { marginTop: 0 } }, "Excludes warm-up iterations and separates model inference latency from full end-to-end processing time."),
      h(
        "div",
        { className: "sv-form-row" },
        h(
          "div",
          { className: "sv-field" },
          h("label", null, "Benchmark Mode"),
          h(
            "select",
            { value: mode, onChange: (e) => setMode(e.target.value) },
            h("option", { value: "cpu" }, "CPU (ExecuTorch XNNPACK)"),
            h("option", { value: "npu", disabled: !npuReady }, npuReady ? "NPU (Hailo-8 HailoRT)" : "NPU (Unavailable)"),
            h("option", { value: "compare", disabled: !bothReady }, bothReady ? "Compare Both (CPU vs NPU)" : "Compare Both (Unavailable)")
          )
        ),
        h(
          "div",
          { className: "sv-field" },
          h("label", null, "Test Frame Source"),
          h(
            "select",
            { value: inspId, onChange: (e) => setInspId(e.target.value) },
            h("option", { value: "" }, "Bundled sample road fixture"),
            completedInsps.map((it) => h("option", { key: it.inspection_id, value: it.inspection_id }, `${it.inspection_id} (${(it.filenames || [])[0] || it.input_type})`))
          )
        ),
        h(
          "div",
          { className: "sv-field", style: { minWidth: 100 } },
          h("label", null, "Warmup runs"),
          h("input", { type: "number", min: 0, max: 20, value: warmup, onChange: (e) => setWarmup(Number(e.target.value)) })
        ),
        h(
          "div",
          { className: "sv-field", style: { minWidth: 100 } },
          h("label", null, "Measured runs"),
          h("input", { type: "number", min: 1, max: 100, value: measured, onChange: (e) => setMeasured(Number(e.target.value)) })
        ),
        h(
          "div",
          { className: "sv-field", style: { minWidth: 110 } },
          h("label", null, "Conf. threshold"),
          h("input", { type: "number", step: "0.05", min: "0.10", max: "0.90", value: confThres, onChange: (e) => setConfThres(Number(e.target.value)) })
        ),
        h("button", { type: "button", className: "btn", disabled: busy, onClick: runBenchmark }, busy ? "Benchmarking..." : "Run Benchmark")
      ),
      errMsg ? h("div", { className: "alert", style: { marginTop: 12 } }, errMsg) : null,
      job && job.status === "completed" ? h("div", { className: "alert ok", style: { marginTop: 12 } }, `Benchmark completed (${job.report.measured_runs} measured runs, duration ${job.report.duration_sec}s).`) : null
    ),

    // Measured Results Card (show only actually measured engine results — no placeholder NPU rows when unmeasured)
    rep
      ? h(
          "div",
          { className: "card", style: { marginTop: 16 } },
          h(
            "div",
            { className: "ch" },
            h("h2", null, rep.npu && rep.npu.processed_frames > 0 ? "CPU vs Hailo-8 NPU Comparison" : "Measured CPU Benchmark Results"),
            rep.csv_available && rep.csv_url
              ? h("a", { className: "btn ghost sv-btn-sm", href: `/api/v1${rep.csv_url.replace(/^\/api\/v1/, "")}`, download: "cpu_npu_comparison_latest.csv" }, "Download CSV")
              : null
          ),
          h(
            "div",
            { className: "scroll" },
            h(
              "table",
              null,
              h("thead", null, h("tr", null, h("th", null, "Metric"), h("th", null, "CPU (ExecuTorch)"), rep.npu && rep.npu.processed_frames > 0 ? h("th", null, "Hailo-8 NPU") : null)),
              h(
                "tbody",
                null,
                h("tr", null, h("td", null, "Processed / Failed Frames"), h("td", null, `${rep.cpu.processed_frames} / ${rep.cpu.failed_frames}`), rep.npu && rep.npu.processed_frames > 0 ? h("td", null, `${rep.npu.processed_frames} / ${rep.npu.failed_frames}`) : null),
                h("tr", null, h("td", null, "Mean / Median / p95 Inference"), h("td", null, `${rep.cpu.mean_inference_ms} / ${rep.cpu.median_inference_ms} / ${rep.cpu.p95_inference_ms} ms`), rep.npu && rep.npu.processed_frames > 0 ? h("td", null, `${rep.npu.mean_inference_ms} / ${rep.npu.median_inference_ms} / ${rep.npu.p95_inference_ms} ms`) : null),
                h("tr", null, h("td", null, "Inference Throughput"), h("td", null, `${rep.cpu.inference_fps} FPS`), rep.npu && rep.npu.processed_frames > 0 ? h("td", null, `${rep.npu.inference_fps} FPS`) : null),
                h("tr", null, h("td", null, "Mean End-to-End Time & FPS"), h("td", null, `${rep.cpu.mean_total_ms} ms (${rep.cpu.end_to_end_fps} FPS)`), rep.npu && rep.npu.processed_frames > 0 ? h("td", null, `${rep.npu.mean_total_ms} ms (${rep.npu.end_to_end_fps} FPS)`) : null),
                h("tr", null, h("td", null, "Total Detections / Duration"), h("td", null, `${((rep.accuracy_comparison || {}).cpu || {}).total_detections ?? 0} detections · ${rep.duration_sec}s`), rep.npu && rep.npu.processed_frames > 0 ? h("td", null, `${((rep.accuracy_comparison || {}).npu || {}).total_detections ?? 0} detections`) : null)
              )
            )
          )
        )
      : null
  );
}

/* ============================================================================
 * 4. COMPACT REPORTS AND SETTINGS PAGE (Matches 4-Column Reference Screenshot)
 * ========================================================================== */
function CompactZonesSvg({ config }) {
  if (!config || !Array.isArray(config.zones) || config.zones.length === 0) {
    return h("div", { className: "alert" }, "Zone configuration not loaded.");
  }
  const w = 320, ht = 180;
  return h(
    "svg",
    { className: "frame", viewBox: `0 0 ${w} ${ht}`, role: "img", "aria-label": "Configured junction zone polygons", style: { background: "#1d2939" } },
    h("rect", { width: w, height: ht, fill: "#1d2939" }),
    config.zones.map((z, idx) => {
      const pal = zonePaletteFor(z);
      const pts = z.polygon_norm || z.points || [];
      const extra = z.extra_polygons_norm || [];
      const polys = [pts, ...extra].filter((p) => Array.isArray(p) && p.length >= 3);
      const anchor = (pts && pts[0]) || [0.05, 0.1];
      return h(
        "g",
        { key: z.zone_id || z.name || idx },
        polys.map((poly, pi) =>
          h("polygon", {
            key: pi,
            points: poly.map(([x, y]) => `${x * w},${y * ht}`).join(" "),
            fill: pal.fill,
            fillOpacity: 0.34,
            stroke: pal.stroke,
            strokeWidth: 1.4,
          })
        ),
        h(
          "text",
          {
            x: Math.min(w - 70, Math.max(6, anchor[0] * w + 6)),
            y: Math.min(ht - 8, Math.max(14, anchor[1] * ht + 14)),
            fontSize: 9.5,
            fontWeight: 700,
            fill: "#ffffff",
          },
          String(z.name || "").toUpperCase()
        )
      );
    })
  );
}

export function ReportsAndSettingsPage({
  health,
  safetyConfig,
  zonesConfig,
  results,
  report,
  benchmark,
  activeSceneId,
  onChangeActiveScene,
  onRefreshZones,
  onDownloadCsv,
  inspections,
}) {
  const { React, apiFetch } = getHost();
  const [editorOpen, setEditorOpen] = React.useState(false);
  const [resetBusy, setResetBusy] = React.useState(false);
  const [zoneMsg, setZoneMsg] = React.useState(null);

  const fmt = (val, suffix = "") => (val === null || val === undefined || Number.isNaN(val) ? "Not measured yet" : `${val}${suffix}`);

  const handleResetDefault = async () => {
    setResetBusy(true);
    setZoneMsg(null);
    try {
      await apiFetch(`/zones/reset?scene_id=${encodeURIComponent(activeSceneId || "default")}`, { method: "POST" });
      if (onRefreshZones) await onRefreshZones(activeSceneId || "default");
      setZoneMsg({ ok: true, text: `Reset zones for '${activeSceneId || "default"}' to default starter polygons.` });
    } catch (e) {
      setZoneMsg({ ok: false, text: e instanceof Error ? e.message : "Reset failed" });
    } finally {
      setResetBusy(false);
    }
  };

  const zonesList = (zonesConfig && zonesConfig.zones) || [];
  const sceneName = (zonesConfig && zonesConfig.scene_id) || activeSceneId || "default";

  return h(
    "div",
    null,
    h("h1", null, "Reports and settings"),
    h("p", { className: "sub" }, "Offline Safety Brief, device benchmark metrics, system status, and editable junction zones."),

    h(
      "div",
      { className: "sv-reports-grid" },
      // Card 1 — Safety Brief
      h(
        "div",
        { className: "card" },
        h(
          "div",
          { className: "ch" },
          h("h2", null, "Safety Brief"),
          results
            ? h("button", { type: "button", className: "btn ghost sv-btn-sm", onClick: () => onDownloadCsv(results.inspection_id) }, "Download CSV")
            : null
        ),
        report && report.available && report.safety_brief
          ? h(
              "div",
              null,
              h("p", { style: { marginTop: 0, fontSize: 13.5 } }, report.safety_brief),
              h(
                "dl",
                null,
                h("dt", null, "Frames processed"),
                h("dd", null, report.frames_processed ?? (results && results.total_frames) ?? "Not measured yet"),
                h("dt", null, "Peak risk frame"),
                h("dd", null, `Frame ${report.peak_risk_frame ?? "-"}`),
                h("dt", null, "Co-presence frames"),
                h("dd", null, report.co_presence_frame_percentage !== undefined ? `${report.co_presence_frame_percentage}%` : "Not measured yet"),
                h("dt", null, "Busiest zone"),
                h("dd", null, report.busiest_zone ?? "Not measured yet"),
                h("dt", null, "Inference throughput"),
                h("dd", null, report.approximate_fps !== undefined ? `${report.approximate_fps} FPS (${report.average_inference_ms} ms)` : "Not measured yet")
              )
            )
          : h("p", { className: "note", style: { margin: 0 } }, "Run or select an inspection with Generate offline Safety Brief enabled to view the written operational summary.")
      ),

      // Card 2 — Performance Benchmark
      h(
        "div",
        { className: "card" },
        h("h2", null, "Performance Benchmark"),
        h(
          "dl",
          null,
          h("dt", null, "Mean inference latency"),
          h("dd", null, fmt(benchmark && benchmark.mean_inference_ms, " ms")),
          h("dt", null, "Median inference latency"),
          h("dd", null, fmt(benchmark && benchmark.median_inference_ms, " ms")),
          h("dt", null, "p95 inference latency"),
          h("dd", null, fmt(benchmark && benchmark.p95_inference_ms, " ms")),
          h("dt", null, "Average inference FPS"),
          h("dd", null, fmt(benchmark && benchmark.average_fps, " FPS")),
          h("dt", null, "Mean end-to-end latency"),
          h("dd", null, fmt(benchmark && benchmark.mean_total_ms, " ms")),
          h("dt", null, "End-to-end FPS"),
          h("dd", null, fmt(benchmark && benchmark.end_to_end_fps, " FPS")),
          h("dt", null, "Process RAM"),
          h("dd", null, fmt(benchmark && benchmark.ram_usage_mb, " MB")),
          benchmark && benchmark.cpu_temperature_c != null
            ? [h("dt", { key: "ct1" }, "CPU temperature"), h("dd", { key: "ct2" }, `${benchmark.cpu_temperature_c} °C`)]
            : null,
          benchmark && benchmark.throttling_detected != null
            ? [h("dt", { key: "th1" }, "Thermal throttling"), h("dd", { key: "th2" }, benchmark.throttling_detected ? "Yes" : "No")]
            : null
        ),
        h(
          "p",
          { className: "note", style: { marginTop: 12, marginBottom: 0 } },
          benchmark && benchmark.is_raspberry_pi
            ? `Measured on Raspberry Pi hardware (${benchmark.hardware}).`
            : benchmark
            ? `Measured locally on ${benchmark.hardware} (not Raspberry Pi 5 hardware).`
            : "No live benchmark loaded yet; run an inspection to capture host measurements."
        )
      ),

      // Card 3 — System & Privacy
      h(
        "div",
        { className: "card" },
        h("h2", null, "System & Privacy"),
        health
          ? h(
              "dl",
              null,
              h("dt", null, "Backend status"),
              h("dd", null, health.status),
              h("dt", null, "Operation mode"),
              h("dd", null, health.mode || "offline"),
              h("dt", null, "Host architecture"),
              h("dd", null, `${health.device} (${health.cpu_architecture})`),
              h("dt", null, "Active model"),
              h("dd", null, `${health.model.name} ${health.model.quantization} (${health.model.size_mb} MB · ${health.model.loaded ? "loaded" : "not loaded"})`),
              h("dt", null, "Runtime"),
              h("dd", null, `${health.runtime.name} + ${health.runtime.backend}`),
              h("dt", null, `Trained classes (${health.model.classes.length})`),
              h("dd", null, health.model.classes.join(", ")),
              h("dt", null, "Default person blurring"),
              h("dd", null, safetyConfig && safetyConfig.privacy_blur_persons ? "Enabled" : "Disabled")
            )
          : h("div", { className: "alert" }, "Backend not reachable.")
      ),

      // Card 4 — Zones Configuration
      h(
        "div",
        { className: "card" },
        h(
          "div",
          { className: "ch" },
          h("h2", null, "Zones Configuration"),
          h(
            "button",
            { type: "button", className: "btn ghost sv-btn-sm", onClick: () => setEditorOpen(true) },
            "Edit polygons"
          )
        ),
        h(CompactZonesSvg, { config: zonesConfig }),
        h(
          "div",
          { className: "tag-list" },
          h("span", { className: "tag" }, `Scene: ${sceneName}`),
          h("span", { className: "tag" }, `Zones: ${zonesList.length}`),
          h("span", { className: "tag" }, `Anchor: ${(zonesConfig && zonesConfig.anchor_mode) || "bottom_center"}`),
          zonesConfig && zonesConfig.calibration && zonesConfig.calibration.calibrated
            ? h("span", { className: "tag", style: { background: "#ecfdf3", color: "#027a48" } }, "Ground calibrated")
            : null,
          zonesList.map((z, idx) => h("span", { className: "tag", key: z.zone_id || idx }, z.name))
        ),
        h(
          "div",
          { style: { marginTop: 10, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" } },
          h("button", { type: "button", className: "btn ghost sv-btn-sm", disabled: resetBusy, onClick: handleResetDefault }, resetBusy ? "Resetting..." : "Reset to default")
        ),
        zoneMsg ? h("div", { className: zoneMsg.ok ? "alert ok" : "alert", style: { marginTop: 10, fontSize: 12.5, padding: "8px 10px" } }, zoneMsg.text) : null,
        h("p", { className: "note", style: { marginTop: 10, marginBottom: 0 } }, "Polygons above reflect the active backend zone configuration used for Road, Crossing, and Footpath occupancy.")
      )
    ),

    // Optional Zone Editor Modal Workspace (opened ONLY on clicking Edit polygons)
    editorOpen
      ? h(ZoneEditorModal, {
          inspections,
          activeSceneId: sceneName,
          onChangeActiveScene,
          onSaved: async (savedSid) => {
            if (onRefreshZones) await onRefreshZones(savedSid);
          },
          onClose: () => setEditorOpen(false),
        })
      : null
  );
}

/* ============================================================================
 * 5. OPTIONAL INTERACTIVE ZONE EDITOR MODAL WORKSPACE (Confidence-Aware Suggestions + Homography)
 * ========================================================================== */
export function ZoneEditorModal({ inspections, activeSceneId, onChangeActiveScene, onSaved, onClose }) {
  const { React, apiFetch, mediaUrl } = getHost();

  const [sceneId, setSceneId] = React.useState(activeSceneId || "default");
  const [previewMeta, setPreviewMeta] = React.useState(null);
  const [zones, setZones] = React.useState([]);
  const [savedSnapshot, setSavedSnapshot] = React.useState([]);
  const [selectedIdx, setSelectedIdx] = React.useState(0);
  const [selectedVertexIdx, setSelectedVertexIdx] = React.useState(null);

  const [tool, setTool] = React.useState("edit"); // "edit" | "draw" | "calib"
  const [drawPoints, setDrawPoints] = React.useState([]);
  const [newZoneName, setNewZoneName] = React.useState("Main Road");
  const [newZoneType, setNewZoneType] = React.useState("road");

  const [undoStack, setUndoStack] = React.useState([]);
  const [redoStack, setRedoStack] = React.useState([]);
  const [confirmDeleteIdx, setConfirmDeleteIdx] = React.useState(null);
  const [suggestWarnings, setSuggestWarnings] = React.useState([]);
  const [calibration, setCalibration] = React.useState(null);
  const [calibPts, setCalibPts] = React.useState([]);
  const [calibW, setCalibW] = React.useState(7.0);
  const [calibL, setCalibL] = React.useState(4.0);

  const [busy, setBusy] = React.useState(false);
  const [banner, setBanner] = React.useState(null);

  const svgRef = React.useRef(null);
  const dragRef = React.useRef(null);

  const pushHistory = React.useCallback((prev) => {
    setUndoStack((stk) => [...stk.slice(-29), cloneZones(prev)]);
    setRedoStack([]);
  }, []);

  const loadScene = React.useCallback(
    async (sid) => {
      try {
        const data = await apiFetch(`/zones?scene_id=${encodeURIComponent(sid)}`);
        const list = (data.zones || []).map((z, i) => ({
          zone_id: z.zone_id || `zone_${i + 1}`,
          name: z.name || `Zone ${i + 1}`,
          type: inferZoneTypeJS(z),
          points: z.points || z.polygon_norm || [],
          polygon_norm: z.points || z.polygon_norm || [],
          extra_polygons_norm: z.extra_polygons_norm || undefined,
        }));
        setZones(cloneZones(list));
        setSavedSnapshot(cloneZones(list));
        setSelectedIdx(list.length > 0 ? 0 : -1);
        setSelectedVertexIdx(null);
        setCalibration(data.calibration || null);
        setCalibPts((data.calibration && data.calibration.image_points_norm) || []);
        setUndoStack([]);
        setRedoStack([]);
        setConfirmDeleteIdx(null);
        setSuggestWarnings([]);
      } catch (e) {
        setBanner({ ok: false, text: e instanceof Error ? e.message : "Failed to load zones" });
      }
    },
    [apiFetch]
  );

  React.useEffect(() => {
    loadScene(sceneId);
  }, [sceneId, loadScene]);

  const handleUploadPreview = async (file) => {
    if (!file) return;
    setBusy(true);
    setBanner(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await apiFetch("/zones/preview", { method: "POST", body: fd }, 60000);
      setPreviewMeta(res);
      setSceneId(res.scene_id);
      if (onChangeActiveScene) onChangeActiveScene(res.scene_id);
      setBanner({ ok: true, text: `Loaded representative frame (${res.width}x${res.height}) for scene '${res.scene_id}'.` });
    } catch (e) {
      setBanner({ ok: false, text: e instanceof Error ? e.message : "Preview extraction failed" });
    } finally {
      setBusy(false);
    }
  };

  const handleLoadInspectionPreview = async (iid) => {
    if (!iid) return;
    setBusy(true);
    setBanner(null);
    try {
      const res = await apiFetch(`/inspections/${encodeURIComponent(iid)}/preview`);
      setPreviewMeta(res);
      setSceneId(res.scene_id);
      if (onChangeActiveScene) onChangeActiveScene(res.scene_id);
      setBanner({ ok: true, text: `Loaded frame from ${iid} (${res.width}x${res.height}).` });
    } catch (e) {
      setBanner({ ok: false, text: e instanceof Error ? e.message : "Failed loading inspection frame" });
    } finally {
      setBusy(false);
    }
  };

  const handleAutoSuggest = async () => {
    setBusy(true);
    setBanner(null);
    try {
      const res = await apiFetch(`/zones/suggest?scene_id=${encodeURIComponent(sceneId)}`, { method: "POST" });
      const suggested = res.zones || [];
      pushHistory(zones);
      setZones(cloneZones(suggested));
      setSelectedIdx(suggested.length > 0 ? 0 : -1);
      setSelectedVertexIdx(null);
      setTool("edit");
      setSuggestWarnings(res.warnings || []);
      setBanner({
        ok: true,
        text: `Generated confidence-aware draft polygons (overall confidence ${Math.round((res.overall_confidence || 0) * 100)}%). Review & adjust uncertain areas before saving — suggestions are NOT auto-saved.`,
      });
    } catch (e) {
      setBanner({ ok: false, text: e instanceof Error ? e.message : "Suggestion failed" });
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async () => {
    setBusy(true);
    setBanner(null);
    try {
      const payload = {
        scene_id: sceneId || "default",
        anchor_mode: "bottom_center",
        zones: zones.map((z) => ({
          zone_id: z.zone_id,
          name: z.name,
          type: z.type,
          points: z.points,
          polygon_norm: z.points,
          extra_polygons_norm: z.extra_polygons_norm,
        })),
      };
      const saved = await apiFetch("/zones", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setSavedSnapshot(cloneZones(saved.zones || zones));
      setSuggestWarnings([]);
      if (onChangeActiveScene) onChangeActiveScene(sceneId);
      if (onSaved) await onSaved(sceneId);
      setBanner({ ok: true, text: `Saved ${zones.length} confirmed zone(s) to scene '${sceneId}'.` });
    } catch (e) {
      setBanner({ ok: false, text: e instanceof Error ? e.message : "Failed saving zones" });
    } finally {
      setBusy(false);
    }
  };

  const handleSaveCalibration = async () => {
    if (calibPts.length !== 4) {
      setBanner({ ok: false, text: "Click exactly 4 ground-plane corner points on the preview before saving calibration." });
      return;
    }
    setBusy(true);
    setBanner(null);
    try {
      const worldPts = [
        [0, 0],
        [Number(calibW), 0],
        [Number(calibW), Number(calibL)],
        [0, Number(calibL)],
      ];
      const res = await apiFetch("/zones/calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene_id: sceneId,
          image_points_norm: calibPts,
          world_points_m: worldPts,
          frame_width: (previewMeta && previewMeta.width) || 1280,
          frame_height: (previewMeta && previewMeta.height) || 720,
        }),
      });
      setCalibration(res.calibration || null);
      setTool("edit");
      if (onSaved) await onSaved(sceneId);
      setBanner({ ok: true, text: `Saved 4-point homography calibration (${calibW}m × ${calibL}m, RMSE=${res.calibration.reprojection_rmse_m}m).` });
    } catch (e) {
      setBanner({ ok: false, text: e instanceof Error ? e.message : "Homography calibration failed" });
    } finally {
      setBusy(false);
    }
  };

  const clientToNorm = (e) => {
    const svg = svgRef.current;
    if (!svg) return [0.5, 0.5];
    const r = svg.getBoundingClientRect();
    return [clamp01((e.clientX - r.left) / Math.max(1, r.width)), clamp01((e.clientY - r.top) / Math.max(1, r.height))];
  };

  const handleCanvasPointerDown = (e) => {
    const [nx, ny] = clientToNorm(e);
    if (tool === "calib") {
      if (calibPts.length < 4) setCalibPts((prev) => [...prev, [nx, ny]]);
      return;
    }
    if (tool === "draw") {
      setDrawPoints((prev) => [...prev, [nx, ny]]);
      return;
    }
    const sel = zones[selectedIdx];
    if (sel && pointInPoly([nx, ny], sel.points)) {
      pushHistory(zones);
      dragRef.current = { kind: "poly", zoneIdx: selectedIdx, startX: nx, startY: ny, origPoints: cloneZones(sel.points) };
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }
    for (let i = 0; i < zones.length; i++) {
      if (pointInPoly([nx, ny], zones[i].points)) {
        setSelectedIdx(i);
        setSelectedVertexIdx(null);
        return;
      }
    }
  };

  const handlePointerMove = (e) => {
    const d = dragRef.current;
    if (!d) return;
    const [nx, ny] = clientToNorm(e);
    setZones((prev) => {
      const next = cloneZones(prev);
      const z = next[d.zoneIdx];
      if (!z) return prev;
      if (d.kind === "vertex") {
        z.points[d.vertIdx] = [nx, ny];
        z.polygon_norm = z.points;
      } else if (d.kind === "poly") {
        let dx = nx - d.startX, dy = ny - d.startY;
        const b = polyBounds(d.origPoints);
        dx = Math.max(-b.minX, Math.min(1 - b.maxX, dx));
        dy = Math.max(-b.minY, Math.min(1 - b.maxY, dy));
        z.points = d.origPoints.map(([px, py]) => [clamp01(px + dx), clamp01(py + dy)]);
        z.polygon_norm = z.points;
      }
      return next;
    });
  };

  const handlePointerUp = () => {
    dragRef.current = null;
  };

  const finishDrawing = () => {
    if (drawPoints.length < 3) {
      setBanner({ ok: false, text: "Click at least 3 corners on the canvas before finishing a polygon." });
      return;
    }
    pushHistory(zones);
    const id = `zone_${Date.now().toString(36)}`;
    const nextZ = {
      zone_id: id,
      name: newZoneName.trim() || `Zone ${zones.length + 1}`,
      type: newZoneType,
      points: drawPoints,
      polygon_norm: drawPoints,
    };
    const next = [...zones, nextZ];
    setZones(next);
    setSelectedIdx(next.length - 1);
    setSelectedVertexIdx(null);
    setDrawPoints([]);
    setTool("edit");
  };

  const svgW = (previewMeta && previewMeta.width) || 960;
  const svgH = (previewMeta && previewMeta.height) || 540;
  const selZone = selectedIdx >= 0 && selectedIdx < zones.length ? zones[selectedIdx] : null;
  const previewSrc = previewMeta && previewMeta.preview_url ? `${mediaUrl(previewMeta.preview_url)}?t=${Date.now()}` : null;

  return h(
    "div",
    { className: "sv-modal-backdrop", role: "dialog", "aria-modal": "true", "aria-label": "Zone Editor Workspace" },
    h(
      "div",
      { className: "sv-modal-panel" },
      h(
        "div",
        { className: "sv-modal-head" },
        h(
          "div",
          null,
          h("h2", null, "Junction Zone Editor"),
          h("p", { className: "note", style: { margin: "4px 0 0" } }, "Calibrate normalized [0..1] Road, Crossing, Left Footpath, and Right Footpath polygons for this scene.")
        ),
        h("button", { type: "button", className: "btn ghost sv-btn-sm", onClick: onClose }, "Close")
      ),

      // Top media + scene bar
      h(
        "div",
        { className: "sv-form-row", style: { marginTop: 0, marginBottom: 14 } },
        h(
          "div",
          { className: "sv-field" },
          h("label", null, "Representative Image / Video"),
          h("input", { type: "file", accept: ".jpg,.jpeg,.png,.webp,.bmp,.mp4,.avi,.mov,.mkv,.m4v", onChange: (e) => handleUploadPreview(e.target.files && e.target.files[0]) })
        ),
        h(
          "div",
          { className: "sv-field" },
          h("label", null, "Or Existing Inspection Frame"),
          h(
            "select",
            { value: "", onChange: (e) => handleLoadInspectionPreview(e.target.value) },
            h("option", { value: "" }, "Select inspection..."),
            (inspections || []).map((it) => h("option", { key: it.inspection_id, value: it.inspection_id }, `${it.inspection_id} (${(it.filenames || [])[0] || it.input_type})`))
          )
        ),
        h(
          "div",
          { className: "sv-field" },
          h("label", null, "Scene ID"),
          h("input", {
            type: "text",
            value: sceneId,
            onChange: (e) => {
              setSceneId(e.target.value);
              if (onChangeActiveScene) onChangeActiveScene(e.target.value || "default");
            },
          })
        ),
        h("button", { type: "button", className: "btn ghost sv-btn-sm", disabled: busy, onClick: handleAutoSuggest }, "Auto-Suggest Zones")
      ),

      banner ? h("div", { className: banner.ok ? "alert ok" : "alert", style: { marginBottom: 12, padding: "8px 12px", fontSize: 13 } }, banner.text) : null,
      suggestWarnings.length > 0
        ? h(
            "div",
            { className: "alert", style: { marginBottom: 12, padding: "8px 12px", fontSize: 13 } },
            h("strong", null, "Manual Verification Needed:"),
            suggestWarnings.map((w, idx) => h("div", { key: idx, style: { marginTop: 3 } }, `• ${w}`))
          )
        : null,

      h(
        "div",
        { className: "sv-ze-layout" },
        // Left: Canvas + Toolbar
        h(
          "div",
          null,
          h(
            "div",
            { className: "sv-ze-toolbar" },
            h("button", { type: "button", className: `sv-btn-sm btn ${tool === "edit" ? "" : "ghost"}`, onClick: () => { setTool("edit"); setDrawPoints([]); } }, "Select / Edit"),
            h("button", { type: "button", className: `sv-btn-sm btn ${tool === "draw" ? "" : "ghost"}`, onClick: () => { setTool("draw"); setDrawPoints([]); } }, "Draw Polygon"),
            tool === "draw"
              ? h("button", { type: "button", className: "sv-btn-sm btn", onClick: finishDrawing }, `Finish (${drawPoints.length} pts)`)
              : null,
            h("button", { type: "button", className: "sv-btn-sm btn ghost", disabled: undoStack.length === 0, onClick: () => { const prev = undoStack[undoStack.length - 1]; setUndoStack((s) => s.slice(0, -1)); setRedoStack((r) => [...r, cloneZones(zones)]); setZones(prev); } }, "Undo"),
            h("button", { type: "button", className: "sv-btn-sm btn ghost", disabled: redoStack.length === 0, onClick: () => { const nxt = redoStack[redoStack.length - 1]; setRedoStack((r) => r.slice(0, -1)); setUndoStack((s) => [...s, cloneZones(zones)]); setZones(nxt); } }, "Redo"),
            h("button", {
              type: "button",
              className: "sv-btn-sm btn ghost",
              disabled: !selZone || selectedVertexIdx === null || selZone.points.length <= 3,
              onClick: () => {
                if (!selZone || selectedVertexIdx === null || selZone.points.length <= 3) return;
                pushHistory(zones);
                setZones((prev) => {
                  const next = cloneZones(prev);
                  next[selectedIdx].points.splice(selectedVertexIdx, 1);
                  next[selectedIdx].polygon_norm = next[selectedIdx].points;
                  return next;
                });
                setSelectedVertexIdx(null);
              },
            }, "Delete Vertex"),
            h("button", { type: "button", className: `sv-btn-sm btn ${tool === "calib" ? "" : "ghost"}`, onClick: () => setTool(tool === "calib" ? "edit" : "calib") }, "Perspective Calib (4-pt)")
          ),

          tool === "calib"
            ? h(
                "div",
                { className: "alert info", style: { marginBottom: 10, padding: "8px 12px", fontSize: 12.5 } },
                h("div", null, `Click 4 corners on the ground rectangle (clockwise: TL, TR, BR, BL). Selected: ${calibPts.length}/4`),
                h(
                  "div",
                  { className: "sv-form-row", style: { marginTop: 8 } },
                  h("div", { className: "sv-field", style: { minWidth: 90 } }, h("label", null, "Width (m)"), h("input", { type: "number", step: "0.5", min: "1", value: calibW, onChange: (e) => setCalibW(e.target.value) })),
                  h("div", { className: "sv-field", style: { minWidth: 90 } }, h("label", null, "Length (m)"), h("input", { type: "number", step: "0.5", min: "1", value: calibL, onChange: (e) => setCalibL(e.target.value) })),
                  h("button", { type: "button", className: "btn sv-btn-sm", disabled: calibPts.length !== 4 || busy, onClick: handleSaveCalibration }, "Save Homography"),
                  h("button", { type: "button", className: "btn ghost sv-btn-sm", onClick: () => setCalibPts([]) }, "Clear Points")
                )
              )
            : null,

          h(
            "div",
            { className: "sv-ze-canvas-wrap" },
            h(
              "svg",
              {
                ref: svgRef,
                className: "sv-ze-svg",
                viewBox: `0 0 ${svgW} ${svgH}`,
                onPointerDown: handleCanvasPointerDown,
                onPointerMove: handlePointerMove,
                onPointerUp: handlePointerUp,
              },
              previewSrc
                ? h("image", { href: previewSrc, x: 0, y: 0, width: svgW, height: svgH, preserveAspectRatio: "none" })
                : [
                    h("rect", { key: "bg", width: svgW, height: svgH, fill: "#101828" }),
                    h("text", { key: "msg", x: svgW / 2, y: svgH / 2, textAnchor: "middle", fill: "#98a2b3", fontSize: 15 }, "Upload an image or video frame above to align zones visually"),
                  ],
              zones.map((z, zi) => {
                const pal = zonePaletteFor(z);
                const isSel = zi === selectedIdx;
                const ptsStr = z.points.map(([px, py]) => `${px * svgW},${py * svgH}`).join(" ");
                const first = z.points[0] || [0.1, 0.1];
                return h(
                  "g",
                  { key: z.zone_id || zi },
                  h("polygon", {
                    points: ptsStr,
                    fill: pal.fill,
                    fillOpacity: isSel ? 0.36 : 0.22,
                    stroke: isSel ? "#ffffff" : pal.stroke,
                    strokeWidth: isSel ? 2.4 : 1.5,
                    strokeDasharray: z.uncertain ? "6,4" : undefined,
                  }),
                  h("text", { x: first[0] * svgW + 6, y: Math.max(18, first[1] * svgH + 16), fill: "#ffffff", fontSize: 13, fontWeight: 700 }, `${z.name} (${z.type})`),
                  isSel && tool === "edit"
                    ? z.points.map(([px, py], vi) => {
                        const nxt = z.points[(vi + 1) % z.points.length];
                        const mx = ((px + nxt[0]) * 0.5) * svgW;
                        const my = ((py + nxt[1]) * 0.5) * svgH;
                        return [
                          h("circle", {
                            key: `m-${vi}`,
                            cx: mx,
                            cy: my,
                            r: 4.5,
                            fill: "#ffffff",
                            fillOpacity: 0.75,
                            stroke: "#475467",
                            strokeWidth: 1,
                            style: { cursor: "copy" },
                            onPointerDown: (e) => {
                              e.stopPropagation();
                              pushHistory(zones);
                              setZones((prev) => {
                                const next = cloneZones(prev);
                                next[zi].points.splice(vi + 1, 0, [clamp01((px + nxt[0]) * 0.5), clamp01((py + nxt[1]) * 0.5)]);
                                next[zi].polygon_norm = next[zi].points;
                                return next;
                              });
                              setSelectedVertexIdx(vi + 1);
                            },
                          }),
                          h("circle", {
                            key: `v-${vi}`,
                            cx: px * svgW,
                            cy: py * svgH,
                            r: selectedVertexIdx === vi ? 7.5 : 6,
                            fill: selectedVertexIdx === vi ? "#f04438" : "#7f56d9",
                            stroke: "#ffffff",
                            strokeWidth: 2,
                            style: { cursor: "grab" },
                            onPointerDown: (e) => {
                              e.stopPropagation();
                              setSelectedVertexIdx(vi);
                              pushHistory(zones);
                              dragRef.current = { kind: "vertex", zoneIdx: zi, vertIdx: vi };
                              svgRef.current.setPointerCapture(e.pointerId);
                            },
                          }),
                        ];
                      })
                    : null
                );
              }),
              drawPoints.length > 0
                ? h(
                    "g",
                    null,
                    h("polyline", { points: drawPoints.map(([px, py]) => `${px * svgW},${py * svgH}`).join(" "), fill: "none", stroke: "#f9fafb", strokeWidth: 2.2, strokeDasharray: "5,4" }),
                    drawPoints.map(([px, py], i) => h("circle", { key: i, cx: px * svgW, cy: py * svgH, r: 5, fill: "#7f56d9", stroke: "#fff", strokeWidth: 1.5 }))
                  )
                : null,
              calibPts.map(([px, py], i) =>
                h(
                  "g",
                  { key: `c-${i}` },
                  h("circle", { cx: px * svgW, cy: py * svgH, r: 6, fill: "#12b76a", stroke: "#fff", strokeWidth: 2 }),
                  h("text", { x: px * svgW + 8, y: py * svgH - 6, fill: "#ecfdf3", fontSize: 12, fontWeight: 700 }, `P${i + 1}`)
                )
              )
            )
          ),
          h("p", { className: "note", style: { marginTop: 8, marginBottom: 0 } }, "Drag inside polygon to move, drag purple vertices to reshape, click white midpoints to add a vertex. Saved normalized [0..1].")
        ),

        // Right: Zone List + Inspector + Save/Reset
        h(
          "div",
          null,
          tool === "draw"
            ? h(
                "div",
                { className: "card", style: { marginBottom: 12, padding: 12 } },
                h("strong", null, "New Polygon Properties"),
                h(
                  "div",
                  { className: "sv-field", style: { marginTop: 8 } },
                  h("label", null, "Zone Name"),
                  h("input", { type: "text", value: newZoneName, onChange: (e) => setNewZoneName(e.target.value) })
                ),
                h(
                  "div",
                  { className: "sv-field", style: { marginTop: 8 } },
                  h("label", null, "Zone Type"),
                  h(
                    "select",
                    { value: newZoneType, onChange: (e) => setNewZoneType(e.target.value) },
                    h("option", { value: "road" }, "road"),
                    h("option", { value: "crossing" }, "crossing"),
                    h("option", { value: "footpath" }, "footpath")
                  )
                )
              )
            : null,

          h("h3", { style: { margin: "0 0 8px", fontSize: 15 } }, `Zones (${zones.length})`),
          h(
            "div",
            { className: "sv-ze-list" },
            zones.map((z, idx) => {
              const pal = zonePaletteFor(z);
              const isSel = idx === selectedIdx;
              return h(
                "div",
                {
                  key: z.zone_id || idx,
                  className: `sv-ze-item ${isSel ? "selected" : ""}`,
                  onClick: () => {
                    setSelectedIdx(idx);
                    setSelectedVertexIdx(null);
                  },
                },
                h(
                  "div",
                  { style: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6, flexWrap: "wrap" } },
                  h("div", { style: { display: "flex", alignItems: "center", gap: 6 } }, h("span", { className: "sv-swatch", style: { background: pal.fill } }), h("strong", { style: { fontSize: 13.5 } }, z.name)),
                  h("span", { className: "note" }, `${z.type} · ${z.points.length} pts`)
                ),
                z.confidence !== undefined
                  ? h(
                      "div",
                      { style: { marginTop: 5, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" } },
                      h("span", { className: `sv-conf-pill ${z.confidence_level || "low"}` }, `${Math.round(z.confidence * 100)}% ${z.confidence_level || "low"}${z.uncertain ? " — verify" : ""}`),
                      z.evidence ? h("span", { className: "note", style: { fontSize: 11.5 } }, z.evidence) : null
                    )
                  : null,
                isSel
                  ? h(
                      "div",
                      { style: { marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }, onClick: (e) => e.stopPropagation() },
                      h("input", {
                        type: "text",
                        value: z.name,
                        style: { padding: "6px 8px", borderRadius: 6, border: "1px solid #d0d5dd" },
                        onChange: (e) =>
                          setZones((prev) => {
                            const next = cloneZones(prev);
                            next[idx].name = e.target.value;
                            return next;
                          }),
                      }),
                      h(
                        "select",
                        {
                          value: z.type,
                          style: { padding: "6px 8px", borderRadius: 6, border: "1px solid #d0d5dd" },
                          onChange: (e) => {
                            pushHistory(zones);
                            setZones((prev) => {
                              const next = cloneZones(prev);
                              next[idx].type = e.target.value;
                              return next;
                            });
                          },
                        },
                        h("option", { value: "road" }, "road"),
                        h("option", { value: "crossing" }, "crossing"),
                        h("option", { value: "footpath" }, "footpath")
                      ),
                      confirmDeleteIdx === idx
                        ? h(
                            "div",
                            { style: { display: "flex", gap: 6 } },
                            h(
                              "button",
                              {
                                type: "button",
                                className: "btn sv-btn-sm",
                                style: { background: "#d92d20" },
                                disabled: zones.length <= 1,
                                onClick: () => {
                                  pushHistory(zones);
                                  const next = cloneZones(zones);
                                  next.splice(idx, 1);
                                  setZones(next);
                                  setSelectedIdx(Math.max(0, next.length - 1));
                                  setConfirmDeleteIdx(null);
                                },
                              },
                              "Confirm Delete"
                            ),
                            h("button", { type: "button", className: "btn ghost sv-btn-sm", onClick: () => setConfirmDeleteIdx(null) }, "Cancel")
                          )
                        : h(
                            "button",
                            {
                              type: "button",
                              className: "btn ghost sv-btn-sm",
                              disabled: zones.length <= 1,
                              onClick: () => setConfirmDeleteIdx(idx),
                            },
                            "Delete Zone"
                          )
                    )
                  : null
              );
            })
          ),

          h(
            "div",
            { style: { marginTop: 14, display: "flex", flexDirection: "column", gap: 8 } },
            h("button", { type: "button", className: "btn", disabled: busy || zones.length === 0, onClick: handleSave }, busy ? "Saving..." : `Save Scene (${sceneId || "default"})`),
            h(
              "button",
              {
                type: "button",
                className: "btn ghost sv-btn-sm",
                onClick: () => {
                  setZones(cloneZones(savedSnapshot));
                  setUndoStack([]);
                  setRedoStack([]);
                  setBanner({ ok: true, text: "Reverted unsaved changes." });
                },
              },
              "Reset Unsaved Changes"
            )
          )
        )
      )
    )
  );
}
