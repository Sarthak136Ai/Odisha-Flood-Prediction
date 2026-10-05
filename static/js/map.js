/**
 * Odisha Flood Intelligence System - Leaflet Geospatial Mapping Module
 * Uses 100% free, non-watermarked, high-performance base tile services with layer control.
 */

let mapInstance = null;

async function initOdishaMap(elementId = "odisha-map", isFullPage = false) {
  const mapEl = document.getElementById(elementId);
  if (!mapEl) return;

  if (mapInstance) {
    mapInstance.remove();
    mapInstance = null;
  }

  // Create Leaflet Map centered on Odisha
  mapInstance = L.map(elementId, {
    center: [20.5, 84.5],
    zoom: isFullPage ? 7 : 6.5,
    zoomControl: true,
    attributionControl: true
  });

  // Base Tile Layers (100% Free, No Watermark, No API Key Required)
  const esriDark = L.layerGroup([
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
      attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
      maxZoom: 16
    }),
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 16
    })
  ]);

  const osmStandard = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
    maxZoom: 19
  });

  const esriTopo = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community",
    maxZoom: 18
  });

  const esriSatellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    maxZoom: 18
  });

  // Default to Esri Dark Canvas (sleek disaster dashboard style)
  esriDark.addTo(mapInstance);

  if (isFullPage) {
    const baseLayers = {
      "🌙 Dark Canvas (Default)": esriDark,
      "🗺️ OpenStreetMap": osmStandard,
      "🏔️ Esri Topographic": esriTopo,
      "🛰️ Satellite Imagery": esriSatellite
    };
    L.control.layers(baseLayers, null, { position: "topright" }).addTo(mapInstance);
  }

  try {
    const res = await fetchJson("/api/district-risk");
    if (!res.districts) return;

    const districts = res.districts;

    districts.forEach(dist => {
      const lat = dist.latitude || 20.5;
      const lon = dist.longitude || 84.5;
      const name = dist.District;
      const floodDays = dist.total_flood_days || 0;
      const rate = dist.flood_frequency_pct || 0;
      const avgRain = dist.avg_annual_rainfall || 0;
      const maxRain = dist.max_single_day_rain || 0;
      const hq = dist.headquarters || name;

      // Color coding based on calibrated 24-year flood frequency
      let color = "#10b981"; // Low (Green)
      let riskLabel = "LOW";
      let radius = 9;

      if (rate >= 3.2 || floodDays >= 3500) {
        color = "#ef4444"; // High (Red)
        riskLabel = "HIGH";
        radius = 15;
      } else if (rate >= 1.8 || floodDays >= 1200) {
        color = "#f59e0b"; // Moderate (Orange)
        riskLabel = "MODERATE";
        radius = 12;
      }

      // Add Circle Marker
      const circle = L.circleMarker([lat, lon], {
        radius: radius,
        fillColor: color,
        color: "#ffffff",
        weight: 1.5,
        opacity: 0.95,
        fillOpacity: 0.8
      }).addTo(mapInstance);

      // Tooltip
      circle.bindTooltip(`<strong>${name}</strong> (${riskLabel} Risk - ${Number(rate).toFixed(1)}%)`, {
        direction: "top",
        offset: [0, -8],
        className: "custom-leaflet-tooltip"
      });

      // Interactive Popup
      const popupHtml = `
        <div style="font-family: 'Inter', sans-serif; min-width: 230px; padding: 2px;">
          <h6 style="margin: 0 0 6px 0; color: #00b4d8; font-weight: 700; border-bottom: 1px solid #334155; padding-bottom: 4px;">
            ${name} District
          </h6>
          <div style="font-size: 11px; color: #94a3b8; margin-bottom: 8px;">
            <strong>HQ:</strong> ${hq}
          </div>
          <table style="width: 100%; font-size: 12px; margin-bottom: 10px; border-collapse: collapse;">
            <tr>
              <td style="color: #94a3b8; padding: 2px 0;">24-Yr Flood Days:</td>
              <td style="font-weight: 700; text-align: right; color: ${color};">${floodDays.toLocaleString()} d</td>
            </tr>
            <tr>
              <td style="color: #94a3b8; padding: 2px 0;">Flood Frequency:</td>
              <td style="font-weight: 600; text-align: right; color: #fff;">${Number(rate).toFixed(2)}%</td>
            </tr>
            <tr>
              <td style="color: #94a3b8; padding: 2px 0;">Avg Annual Rain:</td>
              <td style="font-weight: 600; text-align: right; color: #fff;">${Number(avgRain).toFixed(1)} mm</td>
            </tr>
            <tr>
              <td style="color: #94a3b8; padding: 2px 0;">Peak 1-Day Rain:</td>
              <td style="font-weight: 600; text-align: right; color: #fff;">${Number(maxRain).toFixed(1)} mm</td>
            </tr>
          </table>
          <a href="/drilldown?district=${name}" class="btn btn-sm btn-primary" style="display: block; width: 100%; text-align: center; font-size: 11px; background: #00b4d8; border: none; font-weight: 600; padding: 4px 8px;">
            Inspect District Stations &rarr;
          </a>
        </div>
      `;

      circle.bindPopup(popupHtml);
    });

  } catch (err) {
    console.error("Failed to load map data:", err);
  }
}
