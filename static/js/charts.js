/**
 * Odisha Flood Intelligence System - Chart.js Visualizations Module
 */

function initDashboardCharts(riskData = { low: 114229, mod: 380, high: 1 }) {
  // 1. Historical Rainfall Trend Chart (2001 - 2024 annual extreme rainfall averages)
  const ctxRain = document.getElementById("historicalRainfallChart");
  if (ctxRain) {
    const years = [
      "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008",
      "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016",
      "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"
    ];
    
    // Representative annual extreme rainfall & flood event counts
    const maxRainfall = [
      180, 165, 230, 195, 245, 280, 260, 310, 190, 220,
      340, 250, 360, 275, 230, 210, 290, 315, 385, 410,
      270, 330, 265, 295
    ];
    
    const floodDays = [
      45, 38, 92, 55, 110, 145, 120, 185, 62, 85,
      215, 130, 240, 160, 95, 78, 170, 195, 260, 290,
      140, 210, 115, 165
    ];

    new Chart(ctxRain, {
      type: "line",
      data: {
        labels: years,
        datasets: [
          {
            label: "State Peak 1-Day Rainfall (mm)",
            data: maxRainfall,
            borderColor: "#00b4d8",
            backgroundColor: "rgba(0, 180, 216, 0.15)",
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            yAxisID: "y"
          },
          {
            label: "Total Inundation Station Days",
            data: floodDays,
            borderColor: "#ef4444",
            backgroundColor: "transparent",
            borderWidth: 2,
            borderDash: [4, 4],
            tension: 0.3,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: { color: "#94a3b8", font: { size: 11 } }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b", font: { size: 10 } }
          },
          y: {
            type: "linear",
            display: true,
            position: "left",
            title: { display: true, text: "Peak Rain (mm)", color: "#00b4d8" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            title: { display: true, text: "Flood Days", color: "#ef4444" },
            grid: { drawOnChartArea: false },
            ticks: { color: "#94a3b8" }
          }
        }
      }
    });
  }

  // 2. Risk Distribution Donut Chart
  const ctxRisk = document.getElementById("riskDistributionChart");
  if (ctxRisk) {
    new Chart(ctxRisk, {
      type: "doughnut",
      data: {
        labels: ["Low Risk (<30%)", "Moderate Risk (30-70%)", "High Risk (>=70%)"],
        datasets: [{
          data: [riskData.low, riskData.mod, riskData.high],
          backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
          borderColor: "#0f1c38",
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#94a3b8", padding: 15, font: { size: 11 } }
          },
          tooltip: {
            callbacks: {
              label: (item) => ` ${item.label}: ${item.raw.toLocaleString()} days`
            }
          }
        },
        cutout: "70%"
      }
    });
  }
}
