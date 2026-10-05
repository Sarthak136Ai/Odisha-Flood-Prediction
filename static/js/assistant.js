/**
 * Odisha Flood Intelligence System - AI Assistant Frontend Script
 */

document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatMessages = document.getElementById("chatMessages");
  const promptChips = document.querySelectorAll(".prompt-chip");

  if (!chatForm || !chatInput || !chatMessages) return;

  function appendMessage(text, sender = "bot") {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${sender}`;

    if (sender === "bot") {
      // Basic formatting for Markdown headings/bullets/bold
      let formatted = text
        .replace(/\n\n/g, "<br><br>")
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code class='text-info'>$1</code>")
        .replace(/### (.*?)(<br>|$)/g, "<h6 class='text-info fw-bold mb-2'>$1</h6>")
        .replace(/• (.*?)(<br>|$)/g, "<div class='d-flex align-items-center gap-2 mb-1'><i class='fa-solid fa-circle-dot text-cyan small'></i><span>$1</span></div>");
      bubble.innerHTML = formatted;
    } else {
      bubble.textContent = text;
    }

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendTypingIndicator() {
    const indicator = document.createElement("div");
    indicator.className = "chat-bubble bot typing-indicator-bubble";
    indicator.id = "typingIndicator";
    indicator.innerHTML = `
      <div class="d-flex align-items-center gap-2 text-muted">
        <i class="fa-solid fa-spinner fa-spin"></i>
        <span>Analyzing hydrological models & meteorological stations...</span>
      </div>
    `;
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function removeTypingIndicator() {
    const indicator = document.getElementById("typingIndicator");
    if (indicator) indicator.remove();
  }

  async function sendMessage(query) {
    const text = query.trim();
    if (!text) return;

    appendMessage(text, "user");
    chatInput.value = "";
    appendTypingIndicator();

    try {
      const res = await fetch("/api/assistant/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text })
      });
      const data = await res.json();
      removeTypingIndicator();

      if (data.status === "success") {
        appendMessage(data.response, "bot");
      } else {
        appendMessage(data.response || "Sorry, I could not process your query.", "bot");
      }
    } catch (err) {
      removeTypingIndicator();
      appendMessage("Network or server connection error. Please try again.", "bot");
    }
  }

  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(chatInput.value);
  });

  promptChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-prompt") || chip.textContent.trim();
      sendMessage(query);
    });
  });
});
