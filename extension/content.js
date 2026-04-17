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

    const description =
      document.querySelector('meta[property="og:description"]')?.content ||
      document.querySelector('meta[name="description"]')?.content ||
      "";

    const image =
      document.querySelector('meta[property="og:image"]')?.content ||
      document.querySelector('meta[name="twitter:image"]')?.content ||
      "";

    const text = extractMainText();
    const images = extractImages();

    return {
      title: title?.trim() || "Untitled",
      description: description.trim(),
      image: image.trim(),
      images: images,
      text: text?.trim() || "",
      url: window.location.href,
      wordCount: text ? text.split(/\s+/).length : 0,
    };
  }

  function extractImages() {
    // Collect article images (from article body, then og:image fallback)
    const seen = new Set();
    const images = [];

    // Try article container first
    const containerSelectors = [
      "article", '[role="article"]', ".article-body", ".article-content",
      ".story-body", ".post-content", ".entry-content", "main",
    ];

    let container = null;
    for (const sel of containerSelectors) {
      container = document.querySelector(sel);
      if (container) break;
    }

    const imgEls = container
      ? container.querySelectorAll("img")
      : document.querySelectorAll("article img, main img, .story-body img");

    for (const img of imgEls) {
      const src = img.src || img.dataset.src || "";
      if (!src || seen.has(src)) continue;
      // Skip tiny icons, tracking pixels, avatars
      if (img.naturalWidth > 0 && img.naturalWidth < 80) continue;
      if (img.naturalHeight > 0 && img.naturalHeight < 80) continue;
      if (/icon|logo|avatar|badge|emoji|pixel|tracking/i.test(src)) continue;

      seen.add(src);
      images.push({
        src: src,
        alt: (img.alt || "").trim(),
      });
    }

    // Add og:image if not already captured
    const ogImg = document.querySelector('meta[property="og:image"]')?.content || "";
    if (ogImg && !seen.has(ogImg)) {
      images.unshift({ src: ogImg, alt: "" });
    }

    return images;
  }

  function extractMainText() {
    // Priority-ordered selectors for article content
    const selectors = [
      // Bloomberg
      '[class*="body-content"]',
      '[class*="article-body__content"]',
      '[data-component="body-content"]',
      '[class*="paywall-article"]',
      // General news sites
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

    if (paragraphs.join("\n\n").split(/\s+/).length > 50) {
      return paragraphs.join("\n\n");
    }

    // Last resort: use og:description if nothing else worked (paywalled sites)
    const ogDesc =
      document.querySelector('meta[property="og:description"]')?.content || "";
    return ogDesc;
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
