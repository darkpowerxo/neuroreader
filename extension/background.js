/**
 * NeuroReader Background Service Worker
 * Handles communication between content script and backend API.
 */

const API_BASE = "http://localhost:8421";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyzeText") {
    analyzeText(request.data)
      .then(sendResponse)
      .catch((err) => sendResponse({ error: err.message }));
    return true; // async response
  }

  if (request.action === "checkHealth") {
    checkHealth()
      .then(sendResponse)
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }
});

async function analyzeText(data) {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: data.text,
      url: data.url,
      title: data.title,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return await response.json();
}

async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return await response.json();
}
