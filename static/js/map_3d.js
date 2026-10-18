/**
 * Odisha Flood Intelligence & Early Warning System
 * Interactive 3D Geospatial Flood Risk Atlas (MapLibre GL JS)
 *
 * Implements:
 * - 3D polygon extrusion with vertical scale proportional to flood risk / rainfall
 * - Authentic 30-district GeoJSON boundaries
 * - Dynamic data binding to 2025 operational predictions & historical replay
 * - Smooth pitch/bearing rotation, hover tooltips, click inspection & camera fly-to
 * - Multi-layer switching: Probability, Daily Rain, 7-Day Antecedent, Historical Risk
 * - Side-by-side comparison & Chart.js vulnerability league
 */

let map3d = null;
let currentGeojson = null;
let currentSummary = null;
let selectedDistrictId = null;
let isRotating = false;
let rotationFrameId = null;
let is3DMode = true;
let leagueChart = null;

// Free, non-token basemap style specifications
const BASEMAP_STYLES = {
  "carto-dark": {
    version: 8,
    sources: {
      "base-raster": {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
          "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
          "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png"
        ],
        tileSize: 256,
        attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> &copy; <a href='https://carto.com/'>CARTO</a>"
      }
    },
    layers: [{ id: "base-raster-layer", type: "raster", source: "base-raster" }]
  },
  "carto-light": {
    version: 8,
    sources: {
      "base-raster": {
        type: "raster",
        tiles: [
          "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png",
          "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png",
          "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png"
        ],
        tileSize: 256,
        attribution: "&copy; OpenStreetMap &copy; CARTO"
      }
    },
    layers: [{ id: "base-raster-layer", type: "raster", source: "base-raster" }]
  },
  "osm": {
    version: 8,
    sources: {
      "base-raster": {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "&copy; OpenStreetMap contributors"
      }
    },
    layers: [{ id: "base-raster-layer", type: "raster", source: "base-raster" }]
  },
  "satellite": {
    version: 8,
    sources: {
      "base-raster": {
        type: "raster",
        tiles: [
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ],
        tileSize: 256,
        attribution: "Tiles &copy; Esri, USGS, NOAA"
      }
    },
    layers: [{ id: "base-raster-layer", type: "raster", source: "base-raster" }]
  }
};

/**
 * Initialize 3D MapLibre Instance
 */
function init3DRiskMap(containerId = "odisha-3d-map") {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Initialize MapLibre GL
  map3d = new maplibregl.Map({
    container: containerId,
    style: BASEMAP_STYLES["carto-dark"],
    center: [84.45, 20.45], // Odisha geographic centroid (lon, lat)
    zoom: 6.85,
    pitch: 55, // 3D Perspective angle
    bearing: -12, // Look slightly northwest
    maxPitch: 82,
    minZoom: 5.5,
    maxZoom: 14,
    attributionControl: false
  });

  // Add standard navigation controls (zoom, compass/bearing)
  map3d.addControl(
    new maplibregl.NavigationControl({
      visualizePitch: true,
      showCompass: true,
      showZoom: true
    }),
    "top-right"
  );

  // Add Fullscreen control
  map3d.addControl(new maplibregl.FullscreenControl(), "top-right");

  // Add subtle scale bar
  map3d.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");

  // On Map Load, fetch initial 3D risk data and add layers
  map3d.on("load", () => {
    // Setup 3D directional light for realistic relief and wall shading
    map3d.setLight({
      anchor: "viewport",
      color: "#ffffff",
      intensity: 0.62,
      position: [1.15, 210, 30]
    });

    // Fetch initial dataset
    const dateInput = document.getElementById("datePicker");
    const initialDate = dateInput ? dateInput.value : "2025-08-15";
    const layerSelect = document.getElementById("dataLayerSelect");
    const initialLayer = layerSelect ? layerSelect.value : "probability";

    load3DRiskData(initialDate, initialLayer);
  });

  // Setup Popup Tooltip
  const hoverPopup = new maplibregl.Popup({
    closeButton: false,
    closeOnClick: false,
    offset: 15,
    className: "map-3d-tooltip"
  });

  // Hover Interaction
  map3d.on("mousemove", "odisha-districts-extrusion", (e) => {
    map3d.getCanvas().style.cursor = "pointer";

    if (e.features.length > 0) {
      const feat = e.features[0];
      const p = feat.properties;
      const probPct = (p.predicted_probability * 100).toFixed(1);
      const rf = Number(p.rainfall_mm).toFixed(1);
      const rf7d = Number(p.rainfall_prev_7d_sum).toFixed(1);
      const riskTier = p.risk_level || "LOW";
      const riskColor = p.risk_color || "#10b981";

      const html = `
        <div class="tooltip-3d-card">
          <div class="tooltip-header d-flex justify-content-between align-items-center">
            <strong>${p.display_name || p.district_name}</strong>
            <span class="badge" style="background-color: ${riskColor}; color: #fff; font-size: 10px;">${riskTier}</span>
          </div>
          <div class="tooltip-body mt-1">
            <div class="tooltip-row"><span>HQ:</span> <strong>${p.headquarters || 'N/A'}</strong></div>
            <div class="tooltip-row"><span>Flood Prob:</span> <strong style="color: ${riskColor}; font-size: 13px;">${probPct}%</strong></div>
            <div class="tooltip-row"><span>Daily Rain:</span> <span>${rf} mm</span></div>
            <div class="tooltip-row"><span>7-Day Rain:</span> <span>${rf7d} mm</span></div>
            <div class="tooltip-row"><span>Peak Block:</span> <span>${p.peak_block || 'N/A'}</span></div>
          </div>
          <div class="tooltip-footer text-muted" style="font-size: 9px; margin-top: 4px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 2px;">
            Click polygon for full station telemetry
          </div>
        </div>
      `;

      hoverPopup.setLngLat(e.lngLat).setHTML(html).addTo(map3d);
    }
  });

  map3d.on("mouseleave", "odisha-districts-extrusion", () => {
    map3d.getCanvas().style.cursor = "";
    hoverPopup.remove();
  });

  // Click Interaction: Select District & Fly Camera
  map3d.on("click", "odisha-districts-extrusion", (e) => {
    if (e.features.length > 0) {
      const feat = e.features[0];
      const distName = feat.properties.district_name;
      selectDistrict(distName, true);
    }
  });
}

/**
 * Fetch and Render 3D Risk Data GeoJSON
 */
async function load3DRiskData(dateStr, dataLayer = "probability", filterDistrict = "") {
  showLoading(true);

  try {
    let url = `/api/3d-risk-data?date=${encodeURIComponent(dateStr)}&layer=${encodeURIComponent(dataLayer)}`;
    if (filterDistrict) {
      url += `&district=${encodeURIComponent(filterDistrict)}`;
    }

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
    const data = await resp.json();

    if (data.status !== "success" || !data.geojson) {
      throw new Error(data.message || "Failed to load 3D GeoJSON");
    }

    currentGeojson = data.geojson;
    currentSummary = data.summary;

    // Update or Add GeoJSON source in MapLibre
    if (map3d.getSource("odisha-districts")) {
      map3d.getSource("odisha-districts").setData(currentGeojson);
    } else {
      addGeoJsonLayers(currentGeojson);
    }

    // Update Symbol labels source if present
    updateCentroidLabels(currentGeojson);

    // Update Summary Strip & Badges
    updateSummaryUI(currentSummary);

    // Update League Table & Comparison Chart
    updateLeagueTable(currentSummary.league_table || []);
    renderLeagueChart(currentSummary.league_table || []);

    // Refresh Inspector for current selected district, or default to highest risk
    if (selectedDistrictId) {
      selectDistrict(selectedDistrictId, false);
    } else if (currentSummary.highest_risk_district && currentSummary.highest_risk_district.name) {
      selectDistrict(currentSummary.highest_risk_district.name, false);
    }

  } catch (err) {
    console.error("3D Risk Data Load Error:", err);
    showErrorMessage(`Failed to load 3D flood risk data: ${err.message}`);
  } finally {
    showLoading(false);
  }
}

/**
 * Add Fill-Extrusion, Outline & Label Layers to Map
 */
function addGeoJsonLayers(geojson) {
  // 1. Add Source
  map3d.addSource("odisha-districts", {
    type: "geojson",
    data: geojson
  });

  // 2. Add 3D Fill-Extrusion Layer
  map3d.addLayer({
    id: "odisha-districts-extrusion",
    type: "fill-extrusion",
    source: "odisha-districts",
    paint: {
      "fill-extrusion-color": ["get", "risk_color"],
      "fill-extrusion-height": ["get", "extrusion_height"],
      "fill-extrusion-base": 0,
      "fill-extrusion-opacity": 0.88,
      "fill-extrusion-vertical-gradient": true
    }
  });

  // 3. Add Outline Line Layer for crisp boundaries
  map3d.addLayer({
    id: "odisha-districts-lines",
    type: "line",
    source: "odisha-districts",
    paint: {
      "line-color": "#38bdf8",
      "line-width": 1.4,
      "line-opacity": 0.75
    }
  });

  // 4. Create Centroid Label Points Source
  const centroidFeatures = geojson.features.map(f => ({
    type: "Feature",
    geometry: {
      type: "Point",
      coordinates: [f.properties.centroid_lon, f.properties.centroid_lat]
    },
    properties: {
      district_name: f.properties.district_name,
      display_name: f.properties.display_name,
      probability_pct: f.properties.predicted_probability_pct + "%",
      risk_color: f.properties.risk_color
    }
  }));

  map3d.addSource("district-centroids", {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: centroidFeatures
    }
  });

  // 5. Add Text Label Symbol Layer
  map3d.addLayer({
    id: "district-labels",
    type: "symbol",
    source: "district-centroids",
    layout: {
      "text-field": ["get", "display_name"],
      "text-font": ["Open Sans Bold", "Arial Unicode MS Bold"],
      "text-size": 11,
      "text-offset": [0, 0.6],
      "text-anchor": "top",
      "text-allow-overlap": false
    },
    paint: {
      "text-color": "#ffffff",
      "text-halo-color": "#0b152b",
      "text-halo-width": 2.0
    }
  });
}

/**
 * Update Centroid Labels GeoJSON
 */
function updateCentroidLabels(geojson) {
  if (!map3d.getSource("district-centroids")) return;
  const centroidFeatures = geojson.features.map(f => ({
    type: "Feature",
    geometry: {
      type: "Point",
      coordinates: [f.properties.centroid_lon, f.properties.centroid_lat]
    },
    properties: {
      district_name: f.properties.district_name,
      display_name: f.properties.display_name,
      probability_pct: f.properties.predicted_probability_pct + "%",
      risk_color: f.properties.risk_color
    }
  }));

  map3d.getSource("district-centroids").setData({
    type: "FeatureCollection",
    features: centroidFeatures
  });
}

/**
 * Update Summary Strip Elements
 */
function updateSummaryUI(summary) {
  if (!summary) return;

  const dateBadge = document.getElementById("summaryDateBadge");
  if (dateBadge) dateBadge.textContent = summary.selected_date;

  const horizonBadge = document.getElementById("summaryHorizonBadge");
  if (horizonBadge) horizonBadge.textContent = summary.forecast_horizon;

  const highestName = document.getElementById("summaryHighestDistrict");
  if (highestName && summary.highest_risk_district) {
    highestName.textContent = `${summary.highest_risk_district.name} (${summary.highest_risk_district.probability_pct}%)`;
  }

  const highCountBadge = document.getElementById("summaryHighCount");
  if (highCountBadge && summary.risk_breakdown) {
    highCountBadge.textContent = summary.risk_breakdown.total_elevated;
  }

  const modCountBadge = document.getElementById("summaryModCount");
  if (modCountBadge && summary.risk_breakdown) {
    modCountBadge.textContent = summary.risk_breakdown.moderate;
  }

  const lowCountBadge = document.getElementById("summaryLowCount");
  if (lowCountBadge && summary.risk_breakdown) {
    lowCountBadge.textContent = summary.risk_breakdown.low;
  }

  const avgProbBadge = document.getElementById("summaryAvgProb");
  if (avgProbBadge && summary.state_averages) {
    avgProbBadge.textContent = `${summary.state_averages.mean_probability_pct}%`;
  }

  const sourceBadge = document.getElementById("summarySourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = summary.data_source;
  }
}

/**
 * Select District & Populate Detailed Inspector Panel
 */
function selectDistrict(distName, shouldFly = false) {
  if (!currentGeojson || !currentGeojson.features) return;

  const distClean = distName.trim().toUpperCase();
  const feat = currentGeojson.features.find(f => f.properties.district_name.toUpperCase() === distClean);
  if (!feat) return;

  selectedDistrictId = distClean;
  const p = feat.properties;

  // Sync Dropdown
  const selectElem = document.getElementById("districtSelect");
  if (selectElem && selectElem.value !== distClean) {
    selectElem.value = distClean;
  }

  // Populate Inspector Panel
  const nameEl = document.getElementById("inspectorDistName");
  if (nameEl) nameEl.textContent = p.display_name || p.district_name;

  const hqEl = document.getElementById("inspectorHq");
  if (hqEl) hqEl.textContent = `Headquarters: ${p.headquarters || 'N/A'} (Census Code: ${p.census_code || 'N/A'})`;

  const badgeEl = document.getElementById("inspectorRiskBadge");
  if (badgeEl) {
    badgeEl.textContent = `${p.risk_level} RISK`;
    badgeEl.style.backgroundColor = p.risk_color;
    badgeEl.style.color = "#ffffff";
  }

  const probGauge = document.getElementById("inspectorProbGauge");
  if (probGauge) {
    probGauge.textContent = `${p.predicted_probability_pct}%`;
    probGauge.style.color = p.risk_color;
  }

  const probBar = document.getElementById("inspectorProbBar");
  if (probBar) {
    probBar.style.width = `${Math.min(100, p.predicted_probability_pct)}%`;
    probBar.style.backgroundColor = p.risk_color;
  }

  // Rainfall Telemetry
  setText("inspectorRainDaily", `${p.rainfall_mm} mm`);
  setText("inspectorRain3d", `${p.rainfall_prev_3d_sum} mm`);
  setText("inspectorRain7d", `${p.rainfall_prev_7d_sum} mm`);
  setText("inspectorRain15d", `${p.rainfall_prev_15d_sum} mm`);
  setText("inspectorRain30d", `${p.rainfall_prev_30d_sum} mm`);

  // SHAP Factors
  setText("inspectorFactor1", p.top_risk_driver_1 || "Rainfall_Prev_7d_Sum");
  setText("inspectorFactor2", p.top_risk_driver_2 || "Rainfall_Prev_3d_Sum");
  setText("inspectorFactor3", p.top_risk_driver_3 || "Month_cos");

  // Blocks Information
  setText("inspectorBlocksCount", `${p.blocks_count} reporting block stations`);
  setText("inspectorPeakBlock", p.peak_block || "District Sadar");

  // Sample Blocks Badges
  const blocksContainer = document.getElementById("inspectorBlocksList");
  if (blocksContainer && p.blocks_sample) {
    blocksContainer.innerHTML = p.blocks_sample
      .map(b => `<span class="badge bg-dark border border-secondary text-light me-1 mb-1" style="font-size: 11px;">${b}</span>`)
      .join("");
  }

  // Historical Disaster Frequency
  setText("inspectorHistDays", `${p.historical_flood_days} days`);
  setText("inspectorHistRate", `${p.historical_flood_rate_pct}%`);

  // Drilldown Button Link
  const drilldownBtn = document.getElementById("inspectorDrilldownBtn");
  if (drilldownBtn) {
    drilldownBtn.href = `/drilldown?district=${encodeURIComponent(p.district_name)}`;
  }

  // Camera Fly-To if requested
  if (shouldFly && map3d) {
    map3d.flyTo({
      center: [p.centroid_lon, p.centroid_lat],
      zoom: 7.8,
      pitch: 58,
      bearing: map3d.getBearing(),
      speed: 0.8,
      curve: 1.2
    });
  }
}

/**
 * Update Vulnerability League Table in Drawer
 */
function updateLeagueTable(leagueList) {
  const tableBody = document.querySelector("#leagueTable tbody");
  if (!tableBody) return;

  if (!leagueList || leagueList.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No records available</td></tr>`;
    return;
  }

  tableBody.innerHTML = leagueList.map((item, idx) => {
    let badgeClass = "badge-risk low";
    if (item.risk_level === "VERY HIGH") badgeClass = "badge-risk very-high";
    else if (item.risk_level === "HIGH") badgeClass = "badge-risk high";
    else if (item.risk_level === "MODERATE") badgeClass = "badge-risk moderate";

    return `
      <tr class="cursor-pointer" onclick="selectDistrict('${item.district}', true)">
        <td class="text-muted">${idx + 1}</td>
        <td class="fw-bold text-light">${item.district}</td>
        <td><span class="${badgeClass}" style="font-size: 10px; padding: 2px 6px;">${item.risk_level}</span></td>
        <td class="fw-bold text-cyan">${item.probability_pct}%</td>
        <td>${item.rainfall_mm} mm</td>
        <td class="text-muted small">${item.peak_block}</td>
      </tr>
    `;
  }).join("");
}

/**
 * Render Chart.js Top-8 Vulnerability Chart
 */
function renderLeagueChart(leagueList) {
  const ctx = document.getElementById("leagueChartCanvas");
  if (!ctx) return;

  const topDistricts = leagueList.slice(0, 8);
  const labels = topDistricts.map(d => d.district);
  const probs = topDistricts.map(d => d.probability_pct);
  const rainfalls = topDistricts.map(d => d.rainfall_mm);

  if (leagueChart) {
    leagueChart.destroy();
  }

  leagueChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Predicted Flood Probability (%)",
          data: probs,
          backgroundColor: probs.map(p => p >= 85 ? "#ef4444" : (p >= 70 ? "#f97316" : (p >= 30 ? "#eab308" : "#10b981"))),
          borderColor: "#1e293b",
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: "Daily Rainfall (mm)",
          data: rainfalls,
          type: "line",
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.2)",
          yAxisID: "y1",
          tension: 0.3,
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { size: 11 } }
        },
        tooltip: {
          mode: "index",
          intersect: false
        }
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", font: { size: 10 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y: {
          beginAtZero: true,
          max: 100,
          title: { display: true, text: "Flood Probability (%)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y1: {
          position: "right",
          beginAtZero: true,
          title: { display: true, text: "Rainfall (mm)", color: "#38bdf8" },
          ticks: { color: "#38bdf8" },
          grid: { drawOnChartArea: false }
        }
      }
    }
  });
}

/**
 * Filter by Risk Tier (All, Elevated, High, Moderate, Low)
 */
function filterRiskTiers(tier) {
  if (!map3d || !map3d.getLayer("odisha-districts-extrusion")) return;

  if (tier === "ALL") {
    map3d.setFilter("odisha-districts-extrusion", null);
    map3d.setFilter("odisha-districts-lines", null);
  } else if (tier === "ELEVATED") {
    const filterExp = ["in", ["get", "risk_level"], ["literal", ["HIGH", "VERY HIGH"]]];
    map3d.setFilter("odisha-districts-extrusion", filterExp);
    map3d.setFilter("odisha-districts-lines", filterExp);
  } else {
    const filterExp = ["==", ["get", "risk_level"], tier];
    map3d.setFilter("odisha-districts-extrusion", filterExp);
    map3d.setFilter("odisha-districts-lines", filterExp);
  }
}

/**
 * Switch 3D Extrusion Height Layer
 */
function switchDataLayer(layerName) {
  const dateInput = document.getElementById("datePicker");
  const targetDate = dateInput ? dateInput.value : "2025-08-15";
  load3DRiskData(targetDate, layerName);
}

/**
 * Camera Controls
 */
function resetOdishaView() {
  if (!map3d) return;
  stopAutoRotation();
  map3d.flyTo({
    center: [84.45, 20.45],
    zoom: 6.85,
    pitch: 55,
    bearing: -12,
    speed: 0.8
  });
}

function toggle2D3D() {
  if (!map3d) return;
  is3DMode = !is3DMode;
  const targetPitch = is3DMode ? 55 : 0;
  const btn = document.getElementById("togglePitchBtn");
  if (btn) {
    btn.innerHTML = is3DMode ? `<i class="fa-solid fa-cube me-1"></i> 3D Mode` : `<i class="fa-solid fa-map me-1"></i> 2D Flat`;
  }
  map3d.easeTo({
    pitch: targetPitch,
    duration: 800
  });
}

function toggleAutoRotate() {
  if (isRotating) {
    stopAutoRotation();
  } else {
    startAutoRotation();
  }
}

function startAutoRotation() {
  isRotating = true;
  const btn = document.getElementById("toggleRotateBtn");
  if (btn) btn.classList.add("btn-active-rotate");

  function frame() {
    if (!isRotating) return;
    map3d.setBearing(map3d.getBearing() + 0.15);
    rotationFrameId = requestAnimationFrame(frame);
  }
  rotationFrameId = requestAnimationFrame(frame);
}

function stopAutoRotation() {
  isRotating = false;
  if (rotationFrameId) {
    cancelAnimationFrame(rotationFrameId);
    rotationFrameId = null;
  }
  const btn = document.getElementById("toggleRotateBtn");
  if (btn) btn.classList.remove("btn-active-rotate");
}

function switchBasemap(styleKey) {
  if (!map3d || !BASEMAP_STYLES[styleKey]) return;
  const currentCenter = map3d.getCenter();
  const currentZoom = map3d.getZoom();
  const currentPitch = map3d.getPitch();
  const currentBearing = map3d.getBearing();

  map3d.setStyle(BASEMAP_STYLES[styleKey]);
  map3d.once("styledata", () => {
    map3d.setCenter(currentCenter);
    map3d.setZoom(currentZoom);
    map3d.setPitch(currentPitch);
    map3d.setBearing(currentBearing);

    map3d.setLight({
      anchor: "viewport",
      color: "#ffffff",
      intensity: 0.62,
      position: [1.15, 210, 30]
    });

    if (currentGeojson) {
      addGeoJsonLayers(currentGeojson);
    }
  });
}

/**
 * Compare Two Districts Side-by-Side
 */
function compareDistricts() {
  const d1Select = document.getElementById("compareDist1");
  const d2Select = document.getElementById("compareDist2");
  if (!d1Select || !d2Select || !currentGeojson) return;

  const d1Name = d1Select.value.toUpperCase();
  const d2Name = d2Select.value.toUpperCase();

  const f1 = currentGeojson.features.find(f => f.properties.district_name.toUpperCase() === d1Name);
  const f2 = currentGeojson.features.find(f => f.properties.district_name.toUpperCase() === d2Name);

  const container = document.getElementById("compareResultsBox");
  if (!container) return;

  if (!f1 || !f2) {
    container.innerHTML = `<div class="text-danger small">Could not locate both selected districts.</div>`;
    return;
  }

  const p1 = f1.properties;
  const p2 = f2.properties;

  container.innerHTML = `
    <div class="table-responsive mt-2">
      <table class="table table-sm table-dark border-secondary small">
        <thead>
          <tr>
            <th>Metric</th>
            <th class="text-cyan">${p1.display_name}</th>
            <th class="text-info">${p2.display_name}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Risk Tier</td>
            <td><span class="badge" style="background:${p1.risk_color};">${p1.risk_level}</span></td>
            <td><span class="badge" style="background:${p2.risk_color};">${p2.risk_level}</span></td>
          </tr>
          <tr>
            <td>Flood Probability</td>
            <td class="fw-bold">${p1.predicted_probability_pct}%</td>
            <td class="fw-bold">${p2.predicted_probability_pct}%</td>
          </tr>
          <tr>
            <td>Daily Rainfall</td>
            <td>${p1.rainfall_mm} mm</td>
            <td>${p2.rainfall_mm} mm</td>
          </tr>
          <tr>
            <td>7-Day Cumulative Rain</td>
            <td>${p1.rainfall_prev_7d_sum} mm</td>
            <td>${p2.rainfall_prev_7d_sum} mm</td>
          </tr>
          <tr>
            <td>Highest-Risk Block</td>
            <td>${p1.peak_block}</td>
            <td>${p2.peak_block}</td>
          </tr>
          <tr>
            <td>24-Yr Historical Rate</td>
            <td>${p1.historical_flood_rate_pct}%</td>
            <td>${p2.historical_flood_rate_pct}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  `;
}

/**
 * Utility Helpers
 */
function setText(elemId, text) {
  const el = document.getElementById(elemId);
  if (el) el.textContent = text;
}

function showLoading(isLoading) {
  const spinner = document.getElementById("mapLoadingOverlay");
  if (spinner) {
    spinner.style.display = isLoading ? "flex" : "none";
  }
}

function showErrorMessage(msg) {
  const errBox = document.getElementById("mapAlertBox");
  if (errBox) {
    errBox.textContent = msg;
    errBox.style.display = "block";
    setTimeout(() => { errBox.style.display = "none"; }, 6000);
  }
}
