/**
 * Odisha Flood Intelligence System - Core JavaScript
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Mobile Sidebar Toggle
  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.getElementById("sidebar");
  
  if (menuToggle && sidebar) {
    menuToggle.addEventListener("click", () => {
      sidebar.classList.toggle("show");
    });
  }

  // 2. Live Clock
  const liveClockEl = document.getElementById("liveClock");
  if (liveClockEl) {
    setInterval(() => {
      const now = new Date();
      liveClockEl.textContent = now.toISOString().replace("T", " ").substring(0, 19);
    }, 1000);
  }
});

/**
 * Helper to fetch JSON data with error handling
 */
async function fetchJson(url, options = {}) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || `HTTP Error ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`Fetch error for ${url}:`, err);
    throw err;
  }
}
