# NeuroReader 🧠

**A Chrome extension that reads your brain while you read articles** — powered by Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2) neuroscience foundation model.

NeuroReader extracts article text from any webpage and predicts which brain regions would activate while reading it. The result is an **emotional/cognitive profile** showing the article's impact across 8 neural dimensions: threat, empathy, reward, language, imagery, memory, reasoning, and emotional pain.

## Architecture

```
┌──────────────────┐     ┌─────────────────────────┐
│  Chrome Extension│────▶│  FastAPI Backend (:8421)│
│                  │◀────│                         │
│  • Extract text  │     │  • TRIBE v2 inference   │
│  • Show results  │     │  • Brain region mapping │
│  • Local fallback│     │  • Emotional scoring    │
└──────────────────┘     └─────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │ TRIBE v2 Model   │
                    │ facebook/tribev2)│
                    │ ~1GB, GPU accel  │
                    └──────────────────┘
```

## How It Works

1. **Text extraction** — Content script identifies the main article body using prioritized CSS selectors
2. **TRIBE v2 prediction** — The text is fed through LLaMA 3.2 embeddings → TRIBE v2 transformer → predicted fMRI activations across ~20k cortical vertices
3. **Emotional interpretation** — Predicted activations are mapped to HCP Glasser parcellation ROIs, then scored against 8 neuroscientifically-grounded emotional/cognitive dimensions
4. **Visualization** — Results displayed as a neural activation profile in the extension popup

## Setup

### 1. Backend (requires Python 3.13+, NVIDIA GPU recommended)

```bash
cd backend

# Install all dependencies (including PyTorch cu130 + TRIBE v2)
uv sync

# Add your HuggingFace token (required for LLaMA 3.2 access)
# Copy .env.example to .env and set HF_TOKEN=hf_...
cp .env.example .env

# Start the server
uv run server.py
# or: uvicorn server:app --host 0.0.0.0 --port 8421
```

> **No GPU?** The backend runs in **heuristic mode** automatically if TRIBE v2 can't load. You still get keyword-based emotional analysis — just not the brain model predictions.

### 2. Chrome Extension

1. Open `chrome://extensions/` in Chrome
2. Enable **Developer mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. Navigate to any news article and click the 🧠 icon

## Emotional Dimensions

| Dimension | Brain Regions | What It Detects |
|-----------|--------------|-----------------|
| ⚠️ Threat & Salience | Amygdala, anterior insula | Danger, urgency, alarm |
| 🤝 Empathy & Social | TPJ, STS, mPFC | Understanding others' minds |
| 🎯 Reward & Motivation | Ventral striatum, OFC | Pleasure, desire, incentive |
| 📖 Language & Meaning | Broca's, Wernicke's | Deep comprehension, narrative |
| 🎨 Visual Imagery | V1–V4, fusiform | Mental pictures, scene construction |
| 🧠 Memory & Narrative | Hippocampus, DMN | Personal relevance, story arcs |
| 🔬 Analytical Reasoning | dlPFC, parietal | Critical thinking, evaluation |
| 💔 Emotional Pain | ACC, anterior insula | Suffering, moral distress |

## Modes of Operation

| Mode | When | Quality |
|------|------|---------|
| **TRIBE v2** | Backend running + model loaded + GPU | Gold standard — actual brain model predictions |
| **Heuristic (server)** | Backend running, model failed to load | Keyword-based NLP scoring |
| **Heuristic (local)** | Backend offline | In-browser keyword analysis, no server needed |

## Technical Notes

- TRIBE v2 uses the **"unseen subject" module** — it predicts the average brain response, not a specific individual's
- The text pathway uses **LLaMA 3.2-3B** embeddings fed through TRIBE v2's text-to-speech pipeline (converting text → audio → word timings + text embeddings)
- Brain regions are mapped using the **HCP Glasser parcellation** (360 parcels) and **Harvard-Oxford subcortical atlas** (8 regions)
- The emotional dimensions are derived from established neuroscience literature on functional localization (see paper §2.5–2.6)

## Extending

**Add new emotional dimensions:** Edit `EMOTIONAL_DIMENSIONS` in `backend/server.py`. Each dimension needs a set of HCP parcel names and a description.

**Improve the parcellation mapping:** The current implementation uses a placeholder parcel→vertex mapping. For production accuracy, load the actual Glasser parcellation labels on fsaverage5 using nilearn and map vertices properly.

**Add audio/video support:** TRIBE v2 natively supports all three modalities. The backend could be extended to accept audio URLs (podcasts) or video URLs (YouTube) for full tri-modal brain prediction.

## Credits

- **TRIBE v2**: d'Ascoli et al., Meta FAIR (2026). [Paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) · [Code](https://github.com/facebookresearch/tribev2)
- **Brain parcellation**: Glasser et al. (2016), HCP Multi-modal Parcellation
- **Emotional mapping**: Based on established functional neuroanatomy (Fedorenko et al. 2024, Kanwisher & Yovel 2006, Huth et al. 2016)

## License

Extension code: MIT. TRIBE v2 model: CC-BY-NC 4.0 (Meta).
