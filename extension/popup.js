/**
 * NeuroReader Popup Logic
 * Orchestrates: extract article → call backend → render brain activation profile
 */

const API_BASE = "http://localhost:8421";

const $ = (sel) => document.querySelector(sel);

// ── State management ──────────────────────────────────────────
function showState(id) {
  document.querySelectorAll(".state").forEach((el) => el.classList.add("hidden"));
  $(`#${id}`).classList.remove("hidden");
}

function setLoading(msg) {
  $("#loadingText").textContent = msg;
  showState("loadingState");
}

function setError(title, msg) {
  $("#errorTitle").textContent = title;
  $("#errorMsg").textContent = msg;
  showState("errorState");
}

// ── Main flow ─────────────────────────────────────────────────
async function run() {
  setLoading("Extracting article...");

  try {
    // 1. Check backend health
    const health = await fetchWithTimeout(`${API_BASE}/health`, {}, 3000).then((r) =>
      r.json()
    );
    const badge = $("#modeBadge");
    if (health.model_loaded) {
      badge.textContent = "TRIBE v2";
      badge.classList.add("live");
    } else {
      badge.textContent = "heuristic";
      badge.classList.add("live");
    }
  } catch {
    // Backend not running — try heuristic fallback in-browser
    try {
      const article = await extractArticleFromTab();
      if (!article.text || article.wordCount < 10) {
        setError("No Article Found", "Navigate to an article page and try again.");
        return;
      }
      const result = heuristicAnalysis(article.text, article.images);
      renderResults(article, result);
      $("#modeBadge").textContent = "local";
      $("#modeBadge").classList.add("live");
      return;
    } catch (e2) {
      setError(
        "Backend Offline",
        "Start the NeuroReader server:\nuvicorn server:app --port 8421"
      );
      return;
    }
  }

  // 2. Extract article text from active tab
  setLoading("Reading page content...");
  let article;
  try {
    article = await extractArticleFromTab();
  } catch (e) {
    setError("Extraction Failed", e.message || "Could not read the page content.");
    return;
  }

  if (!article.text || article.wordCount < 10) {
    setError(
      "No Article Found",
      "This page doesn't seem to contain a readable article. Try a news article or blog post."
    );
    return;
  }

  // 3. Send to backend for TRIBE v2 analysis
  setLoading("Predicting brain activation...");
  try {
    const settings = getSettings();
    const response = await fetchWithTimeout(
      `${API_BASE}/analyze`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: article.text,
          url: article.url,
          title: article.title,
          images: settings.images ? (article.images || []).map((i) => i.src) : [],
          enable_audio: settings.audio,
          enable_video: settings.video,
        }),
      },
      120000 // 120s timeout — TTS + WhisperX can be slow
    );

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const result = await response.json();
    renderResults(article, result);
  } catch (e) {
    setError("Analysis Failed", e.message || "The backend returned an error.");
  }
}

// ── Article extraction ────────────────────────────────────────
async function extractArticleFromTab() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]) {
        reject(new Error("No active tab"));
        return;
      }

      const tabId = tabs[0].id;
      const tabUrl = tabs[0].url || "";

      // Can't inject into chrome://, edge://, about:, or extension pages
      if (tabUrl.startsWith("chrome") || tabUrl.startsWith("edge") ||
          tabUrl.startsWith("about:") || tabUrl.startsWith("chrome-extension://")) {
        reject(new Error("Cannot extract from this page type"));
        return;
      }

      chrome.tabs.sendMessage(
        tabId,
        { action: "extractArticle" },
        (response) => {
          if (chrome.runtime.lastError) {
            // Content script not injected — inject it now
            chrome.scripting.executeScript(
              {
                target: { tabId },
                files: ["content.js"],
              },
              (results) => {
                if (chrome.runtime.lastError) {
                  reject(new Error("Cannot access this page"));
                  return;
                }
                setTimeout(() => {
                  chrome.tabs.sendMessage(
                    tabId,
                    { action: "extractArticle" },
                    (res) => {
                      if (chrome.runtime.lastError || !res) {
                        reject(new Error("Could not extract article"));
                      } else {
                        resolve(res);
                      }
                    }
                  );
                }, 300);
              }
            );
          } else if (response) {
            resolve(response);
          } else {
            reject(new Error("Empty response from content script"));
          }
        }
      );
    });
  });
}

// ── Render results ────────────────────────────────────────────
function renderResults(article, result) {
  // Article meta
  $("#articleTitle").textContent = article.title;
  $("#articleDesc").textContent = article.description || "";
  const imgEl = $("#articleImage");
  if (article.image) {
    imgEl.src = article.image;
    imgEl.style.display = "block";
  } else {
    imgEl.style.display = "none";
  }
  $("#articleStats").textContent = `${article.wordCount.toLocaleString()} words · ${result.mode === "tribev2" ? "TRIBE v2 model" : "heuristic analysis"}`;

  // Summary
  $("#summaryBox").textContent = result.summary;

  // Dimensions
  const container = $("#dimensionsContainer");
  container.innerHTML = "";

  // Sort by score descending
  const sorted = Object.entries(result.dimensions).sort(
    (a, b) => b[1].score - a[1].score
  );

  sorted.forEach(([key, dim], i) => {
    const isDominant = key === result.dominant;
    const pct = Math.round(dim.score * 100);

    const row = document.createElement("div");
    row.className = `dim-row${isDominant ? " dominant" : ""}`;
    row.innerHTML = `
      <span class="dim-icon">${dim.icon}</span>
      <div class="dim-info">
        <div class="dim-label">
          ${dim.label}
          ${isDominant ? '<span class="dominant-tag">peak</span>' : ""}
        </div>
        <div class="dim-desc">${dim.description}</div>
      </div>
      <div class="dim-bar-wrap">
        <div class="dim-bar-track">
          <div class="dim-bar-fill" style="background: ${dim.color};" data-width="${pct}%"></div>
        </div>
      </div>
      <span class="dim-score" style="color: ${dim.color};">${pct}</span>
    `;
    container.appendChild(row);
  });

  showState("resultsState");

  // Populate source panel
  const srcImages = $("#sourceImages");
  srcImages.innerHTML = "";
  const allImages = article.images || [];
  if (article.image && !allImages.find((i) => i.src === article.image)) {
    allImages.unshift({ src: article.image, alt: "" });
  }
  allImages.forEach((img) => {
    const el = document.createElement("img");
    el.src = img.src;
    el.alt = img.alt || "";
    el.title = img.alt || "";
    srcImages.appendChild(el);
  });

  $("#sourceText").textContent = article.text || "(no text extracted)";

  // Animate brain SVG hotspots
  const DIM_COLORS = {
    threat_salience: "#E31937",
    empathy_social: "#4A90D9",
    reward_motivation: "#2ECC71",
    language_semantics: "#9B59B6",
    visual_imagery: "#E67E22",
    memory_narrative: "#1ABC9C",
    executive_reasoning: "#3498DB",
    emotional_pain: "#C0392B",
  };

  document.querySelectorAll(".hotspot").forEach((dot) => {
    const dimKey = dot.dataset.dim;
    const score = result.dimensions[dimKey]?.score || 0;
    const color = DIM_COLORS[dimKey] || "#6366f1";
    const radius = 3 + score * 12;
    dot.setAttribute("fill", color);
    dot.setAttribute("fill-opacity", String(0.15 + score * 0.55));
    dot.classList.add("active");
    setTimeout(() => {
      dot.setAttribute("r", String(radius));
    }, 100);
  });

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll(".dim-bar-fill").forEach((bar) => {
        bar.style.width = bar.dataset.width;
      });
    }, 50);
  });
}

// ── In-browser heuristic fallback ─────────────────────────────
function heuristicAnalysis(text, images) {
  const lower = text.toLowerCase();
  const wc = text.split(/\s+/).length;
  const norm = Math.max(wc / 50, 1);

  const kw = (words) =>
    Math.min(words.reduce((s, w) => s + (lower.split(w).length - 1), 0) / norm, 1);

  const imgCount = (images || []).length;

  const scores = {
    threat_salience: kw(["war","attack","kill","threat","danger","crisis","terror","bomb","death","violence","murder","weapon","conflict","fear"]),
    empathy_social: kw(["family","child","mother","community","together","support","help","care","love","people","human","suffer","compassion","refugee"]),
    reward_motivation: kw(["win","success","profit","gain","achieve","reward","opportunity","growth","breakthrough","innovation","celebrate","victory","billion"]),
    language_semantics: Math.min(Math.max(((text.split(/\s+/).reduce((s,w) => s + w.length, 0) / wc) - 4) / 4, 0), 1),
    visual_imagery: Math.min(kw(["image","picture","color","bright","dark","landscape","face","scene","view","beautiful","massive","towering","glow","shadow"]) + Math.min(imgCount * 0.12, 0.4), 1),
    memory_narrative: kw(["remember","story","once","began","journey","years ago","childhood","history","legacy","tradition","memory","past","generation"]),
    executive_reasoning: kw(["analysis","evidence","study","research","data","percent","statistic","finding","conclude","hypothesis","therefore","however"]),
    emotional_pain: kw(["grief","loss","tragic","devastat","heartbreak","mourn","despair","agony","sorrow","betrayal","abandon","lonely","hopeless","injustice"]),
  };

  const mx = Math.max(...Object.values(scores)) || 1;
  for (const k of Object.keys(scores)) {
    scores[k] = Math.round((0.15 + 0.75 * scores[k] / mx) * 1000) / 1000;
  }

  const dims = {
    threat_salience: { label: "Threat & Salience", description: "Amygdala, anterior insula — danger, urgency, alarm", icon: "⚠️", color: "#E31937" },
    empathy_social: { label: "Empathy & Social Cognition", description: "TPJ, STS, mPFC — understanding others' minds", icon: "🤝", color: "#4A90D9" },
    reward_motivation: { label: "Reward & Motivation", description: "Ventral striatum, OFC — pleasure, desire, incentive", icon: "🎯", color: "#2ECC71" },
    language_semantics: { label: "Language & Meaning", description: "Broca's, Wernicke's — comprehension, narrative", icon: "📖", color: "#9B59B6" },
    visual_imagery: { label: "Visual Imagery", description: "Visual cortex, fusiform — mental pictures, scenes", icon: "🎨", color: "#E67E22" },
    memory_narrative: { label: "Memory & Narrative", description: "Hippocampus, DMN — personal relevance, stories", icon: "🧠", color: "#1ABC9C" },
    executive_reasoning: { label: "Analytical Reasoning", description: "dlPFC, parietal — critical thinking, evaluation", icon: "🔬", color: "#3498DB" },
    emotional_pain: { label: "Emotional Pain & Distress", description: "ACC, anterior insula — suffering, moral distress", icon: "💔", color: "#C0392B" },
  };

  const dimensions = {};
  for (const [k, s] of Object.entries(scores)) {
    dimensions[k] = { score: s, ...dims[k] };
  }

  const dominant = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];

  const top3 = Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const parts = top3
    .filter(([, s]) => s > 0.4)
    .map(([k]) => dims[k].label.toLowerCase());
  const summary = parts.length
    ? `This article primarily activates ${parts[0]} processing${parts.length > 1 ? `, with ${parts.slice(1).join(" and ")}` : ""}.`
    : "This article produces a relatively uniform activation pattern.";

  return { dimensions, dominant, summary, mode: "heuristic-local" };
}

// ── Utilities ─────────────────────────────────────────────────
function fetchWithTimeout(url, options = {}, ms = 5000) {
  return Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), ms)
    ),
  ]);
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Load saved settings
  loadSettings();

  run();
  $("#retryBtn").addEventListener("click", run);

  // Settings slide
  $("#settingsBtn").addEventListener("click", () => {
    $("#slider").classList.add("show-settings");
  });
  $("#settingsBack").addEventListener("click", () => {
    $("#slider").classList.remove("show-settings");
  });

  // Persist toggles on change
  ["toggleImages", "toggleAudio", "toggleVideo"].forEach((id) => {
    $(`#${id}`).addEventListener("change", saveSettings);
  });

  // Source panel toggle
  $("#sourceToggle").addEventListener("click", () => {
    const toggle = $("#sourceToggle");
    const panel = $("#sourcePanel");
    toggle.classList.toggle("open");
    panel.classList.toggle("open");
    toggle.querySelector(".arrow").textContent = panel.classList.contains("open") ? "▲" : "▼";
    const label = panel.classList.contains("open") ? "Hide extracted source" : "Show extracted source";
    toggle.childNodes[toggle.childNodes.length - 1].textContent = " " + label;
  });
});

// ── Settings persistence ──────────────────────────────────────
function getSettings() {
  return {
    images: $("#toggleImages").checked,
    audio: $("#toggleAudio").checked,
    video: $("#toggleVideo").checked,
  };
}

function saveSettings() {
  chrome.storage.local.set({ neuroreaderSettings: getSettings() });
}

function loadSettings() {
  chrome.storage.local.get("neuroreaderSettings", (data) => {
    const s = data.neuroreaderSettings || { images: true, audio: true, video: true };
    $("#toggleImages").checked = s.images;
    $("#toggleAudio").checked = s.audio;
    $("#toggleVideo").checked = s.video;
  });
}
