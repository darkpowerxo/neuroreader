/**
 * NeuroReader Content Script
 * Extracts the main article text from the current page.
 * Uses a prioritized selector strategy for common article structures.
 */

(() => {
  // Listen for messages from the popup
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extractArticle") {
      const result = extractArticle();
      sendResponse(result);
    }
    return true; // keep channel open for async
  });

  function extractArticle() {
    const title =
      document.querySelector("h1")?.innerText ||
      document.querySelector('meta[property="og:title"]')?.content ||
      document.title;

    const text = extractMainText();

    return {
      title: title?.trim() || "Untitled",
      text: text?.trim() || "",
      url: window.location.href,
      wordCount: text ? text.split(/\s+/).length : 0,
    };
  }

  function extractMainText() {
    // Priority-ordered selectors for article content
    const selectors = [
      "article",
      '[role="article"]',
      '[itemtype*="Article"]',
      ".post-content",
      ".article-body",
      ".article-content",
      ".entry-content",
      ".story-body",
      ".post-body",
      "#article-body",
      ".content-body",
      ".story-content",
      ".c-entry-content",       // Vox
      ".article__body",         // various
      ".paywall",               // sometimes wraps content
      '[data-testid="article-body"]',
      ".caas-body",             // Yahoo
      "#content",
      "main",
    ];

    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el) {
        const text = extractTextFromElement(el);
        if (text.split(/\s+/).length > 50) {
          return text;
        }
      }
    }

    // Fallback: grab all <p> tags from the page body
    const paragraphs = Array.from(document.querySelectorAll("p"))
      .map((p) => p.innerText.trim())
      .filter((t) => t.length > 40);

    return paragraphs.join("\n\n");
  }

  function extractTextFromElement(el) {
    // Get text from paragraphs within the element, filtering out noise
    const paragraphs = Array.from(el.querySelectorAll("p, h2, h3, blockquote"));

    if (paragraphs.length > 0) {
      return paragraphs
        .map((p) => p.innerText.trim())
        .filter((t) => t.length > 20)
        .join("\n\n");
    }

    // Fallback to full innerText
    return el.innerText;
  }
})();
