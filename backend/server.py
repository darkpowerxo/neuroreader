"""
NeuroReader Backend — TRIBE v2 Emotional Brain Readout Server

Wraps Meta's TRIBE v2 model to predict brain activation patterns from article text,
then maps activations to interpretable emotional/cognitive dimensions.

Requirements:
    pip install fastapi uvicorn tribev2 numpy
    # TRIBE v2: pip install "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8421
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # loads HF_TOKEN (and others) from .env if present

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Brain Region → Emotional/Cognitive Function Mapping
# ---------------------------------------------------------------------------
# Based on HCP Glasser parcellation and Desikan-Killiany atlas.
# Each region maps to functional labels derived from the neuroscience literature
# (see TRIBE v2 paper §2.5–2.6, Fedorenko et al. 2024, Kanwisher & Yovel 2006, etc.)

# Cortical ROI indices on fsaverage5 (approximate vertex ranges for key parcels)
# In practice, we use the Glasser parcellation loaded from nilearn.
# For the MVP, we define functional "zones" by grouping HCP parcels.

EMOTIONAL_DIMENSIONS = {
    "threat_salience": {
        "label": "Threat & Salience",
        "description": "Amygdala, anterior insula — detecting danger, urgency, alarm",
        "icon": "⚠️",
        "color": "#E31937",  # CGI Red — fitting for alarm
        "regions": ["amygdala", "anterior_insula", "ACC"],
        "hcp_parcels": ["FOP1", "FOP2", "FOP3", "AVI", "AAIC", "MI"],
    },
    "empathy_social": {
        "label": "Empathy & Social Cognition",
        "description": "TPJ, STS, medial prefrontal — understanding others' minds",
        "icon": "🤝",
        "color": "#4A90D9",
        "regions": ["TPJ", "STS", "mPFC", "precuneus"],
        "hcp_parcels": ["PGi", "PGs", "STSdp", "STSda", "STSvp", "STSva", "10r", "10v", "9m"],
    },
    "reward_motivation": {
        "label": "Reward & Motivation",
        "description": "Ventral striatum, OFC — pleasure, desire, incentive",
        "icon": "🎯",
        "color": "#2ECC71",
        "regions": ["accumbens", "OFC", "vmPFC"],
        "hcp_parcels": ["OFC", "pOFC", "13l", "11l", "10r"],
    },
    "language_semantics": {
        "label": "Language & Meaning",
        "description": "Broca's area, Wernicke's, angular gyrus — comprehension, narrative",
        "icon": "📖",
        "color": "#9B59B6",
        "regions": ["IFG", "STG", "angular_gyrus", "MTG"],
        "hcp_parcels": ["44", "45", "IFSa", "IFSp", "A5", "STSdp", "TE1a", "PGi", "TPOJ1"],
    },
    "visual_imagery": {
        "label": "Visual Imagery",
        "description": "Visual cortex, fusiform — mental pictures, scene construction",
        "icon": "🎨",
        "color": "#E67E22",
        "regions": ["V1", "V2", "V4", "FFA", "PPA"],
        "hcp_parcels": ["V1", "V2", "V3", "V4", "FFC", "PH", "VVC"],
    },
    "memory_narrative": {
        "label": "Memory & Narrative",
        "description": "Hippocampus, default mode network — personal relevance, story arcs",
        "icon": "🧠",
        "color": "#1ABC9C",
        "regions": ["hippocampus", "retrosplenial", "posterior_cingulate"],
        "hcp_parcels": ["RSC", "POS1", "POS2", "v23ab", "d23ab", "ProS"],
    },
    "executive_reasoning": {
        "label": "Analytical Reasoning",
        "description": "Dorsolateral PFC, parietal — critical thinking, evaluation",
        "icon": "🔬",
        "color": "#3498DB",
        "regions": ["dlPFC", "IPS", "FEF"],
        "hcp_parcels": ["p9-46v", "46", "a9-46v", "IFJa", "IFJp", "IPS1", "AIP"],
    },
    "emotional_pain": {
        "label": "Emotional Pain & Distress",
        "description": "ACC, anterior insula, TPJ — suffering, moral distress",
        "icon": "💔",
        "color": "#C0392B",
        "regions": ["dACC", "anterior_insula", "TPJ"],
        "hcp_parcels": ["a24", "p24", "a32pr", "AAIC", "PGi"],
    },
}

# Subcortical region names matching TRIBE v2's Harvard-Oxford atlas output
SUBCORTICAL_REGIONS = [
    "hippocampus", "lateral_ventricle", "amygdala", "thalamus",
    "caudate", "putamen", "pallidum", "accumbens",
]

# Map subcortical regions to emotional dimensions
SUBCORTICAL_EMOTION_MAP = {
    "amygdala": ["threat_salience"],
    "hippocampus": ["memory_narrative"],
    "accumbens": ["reward_motivation"],
    "thalamus": ["threat_salience", "executive_reasoning"],
    "caudate": ["reward_motivation", "executive_reasoning"],
    "putamen": ["reward_motivation"],
    "pallidum": ["reward_motivation"],
}


# ---------------------------------------------------------------------------
# TRIBE v2 Wrapper
# ---------------------------------------------------------------------------

class TribeWrapper:
    """Wraps TRIBE v2 for text-only inference with emotional interpretation."""

    def __init__(self, cache_folder: str = "./cache"):
        self.cache_folder = Path(cache_folder)
        self.model = None
        self.parcellation = None
        self._load_model()

    def _load_model(self):
        """Load TRIBE v2 from HuggingFace."""
        try:
            from tribev2 import TribeModel
            logging.info("Loading TRIBE v2 from HuggingFace...")
            self.model = TribeModel.from_pretrained(
                "facebook/tribev2",
                cache_folder=self.cache_folder,
            )
            logging.info("TRIBE v2 loaded successfully.")
            self._load_parcellation()
        except ImportError:
            logging.warning(
                "tribev2 not installed. Running in MOCK mode. "
                "Install with: pip install 'tribev2 @ git+https://github.com/facebookresearch/tribev2.git'"
            )
            self.model = None
        except Exception as e:
            logging.warning(f"Failed to load TRIBE v2: {e}. Running in MOCK mode.")
            self.model = None

    def _load_parcellation(self):
        """Load the Glasser parcellation mapper for ROI-based scoring."""
        try:
            from parcellation import GlasserEmotionMapper
            self.parcellation = GlasserEmotionMapper()
            logging.info("Glasser emotion mapper loaded.")
        except Exception as e:
            logging.warning(f"Parcellation mapper unavailable: {e}")
            self.parcellation = None

    def predict_from_text(self, text: str) -> dict:
        """
        Given article text, predict brain activations and return emotional profile.

        Returns dict with:
            - dimensions: {dim_name: {score, label, description, ...}}
            - dominant: the highest-scoring dimension
            - summary: human-readable interpretation
        """
        if self.model is not None:
            return self._predict_real(text)
        else:
            return self._predict_mock(text)

    def _predict_real(self, text: str) -> dict:
        """Run actual TRIBE v2 inference."""
        # TRIBE v2 expects text_path or audio — for text-only, we write a temp file
        # and use the text→speech→transcription pipeline built into TribeModel
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            text_path = f.name

        try:
            # Get events dataframe from text
            df = self.model.get_events_dataframe(text_path=text_path)
            # Predict: returns (n_timesteps, n_vertices) cortical activations
            preds, segments = self.model.predict(events=df)

            # Average across time to get a spatial activation map
            mean_activation = np.mean(np.abs(preds), axis=0)  # (n_vertices,)

            # Map to emotional dimensions
            return self._activations_to_emotions(mean_activation)
        finally:
            os.unlink(text_path)

    def _activations_to_emotions(self, activations: np.ndarray) -> dict:
        """
        Map a (n_vertices,) activation vector to emotional dimension scores.

        Uses the GlasserEmotionMapper for proper ROI-based scoring,
        with a spatial-prior fallback if parcellation data isn't available.
        """
        if self.parcellation is not None:
            scores = self.parcellation.score_dimensions(activations)
        else:
            # Fallback: use spatial priors (imported mapper handles this too)
            try:
                from parcellation import GlasserEmotionMapper
                fallback = GlasserEmotionMapper()
                scores = fallback.score_dimensions(activations)
            except Exception:
                # Last resort: uniform scoring
                scores = {k: 0.5 for k in EMOTIONAL_DIMENSIONS}

        # Round scores
        scores = {k: round(v, 3) for k, v in scores.items()}
        return self._format_output(scores)

    def _predict_mock(self, text: str) -> dict:
        """
        Mock prediction using heuristic NLP analysis.
        Used when TRIBE v2 is not installed — gives a demo experience.
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Heuristic keyword-based scoring (placeholder for real model)
        scores = {}

        # Threat/salience keywords
        threat_words = [
            "war", "attack", "kill", "threat", "danger", "crisis", "terror",
            "bomb", "death", "murder", "weapon", "enemy", "conflict", "fear",
            "catastrophe", "disaster", "emergency", "victim", "violence", "assault",
        ]
        scores["threat_salience"] = min(
            sum(text_lower.count(w) for w in threat_words) / max(word_count / 50, 1),
            1.0,
        )

        # Empathy/social keywords
        empathy_words = [
            "family", "child", "mother", "father", "community", "together",
            "support", "help", "care", "love", "friend", "people", "human",
            "suffer", "compassion", "empathy", "refugee", "homeless", "orphan",
        ]
        scores["empathy_social"] = min(
            sum(text_lower.count(w) for w in empathy_words) / max(word_count / 50, 1),
            1.0,
        )

        # Reward/motivation keywords
        reward_words = [
            "win", "success", "profit", "gain", "achieve", "reward", "opportunity",
            "growth", "breakthrough", "innovation", "celebrate", "victory", "billion",
            "record", "milestone", "launch", "discovery",
        ]
        scores["reward_motivation"] = min(
            sum(text_lower.count(w) for w in reward_words) / max(word_count / 50, 1),
            1.0,
        )

        # Language complexity (sentence length, rare words as proxy)
        avg_word_len = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0
        scores["language_semantics"] = min(max((avg_word_len - 4) / 4, 0), 1.0)

        # Visual imagery keywords
        visual_words = [
            "image", "picture", "color", "bright", "dark", "landscape", "face",
            "scene", "view", "beautiful", "massive", "tiny", "towering", "glow",
            "shadow", "red", "blue", "green", "golden", "silver",
        ]
        scores["visual_imagery"] = min(
            sum(text_lower.count(w) for w in visual_words) / max(word_count / 50, 1),
            1.0,
        )

        # Memory/narrative (story structure indicators)
        narrative_words = [
            "remember", "story", "once", "began", "journey", "years ago",
            "childhood", "history", "legacy", "tradition", "memory", "past",
            "future", "generation", "heritage", "origin",
        ]
        scores["memory_narrative"] = min(
            sum(text_lower.count(w) for w in narrative_words) / max(word_count / 50, 1),
            1.0,
        )

        # Executive reasoning
        reasoning_words = [
            "analysis", "evidence", "study", "research", "data", "percent",
            "statistic", "finding", "conclude", "hypothesis", "argue", "compare",
            "therefore", "however", "furthermore", "nevertheless", "correlation",
        ]
        scores["executive_reasoning"] = min(
            sum(text_lower.count(w) for w in reasoning_words) / max(word_count / 50, 1),
            1.0,
        )

        # Emotional pain
        pain_words = [
            "grief", "loss", "tragic", "devastat", "heartbreak", "mourn",
            "despair", "agony", "sorrow", "betrayal", "abandon", "lonely",
            "depressed", "hopeless", "injustice", "oppression",
        ]
        scores["emotional_pain"] = min(
            sum(text_lower.count(w) for w in pain_words) / max(word_count / 50, 1),
            1.0,
        )

        # Normalize so max is ~0.9 and there's always some baseline activation
        max_score = max(scores.values()) or 1.0
        for k in scores:
            scores[k] = round(0.15 + 0.75 * (scores[k] / max_score), 3)

        return self._format_output(scores)

    def _format_output(self, scores: dict) -> dict:
        """Format scores into the API response structure."""
        dimensions = {}
        for dim_name, score in scores.items():
            info = EMOTIONAL_DIMENSIONS[dim_name]
            dimensions[dim_name] = {
                "score": score,
                "label": info["label"],
                "description": info["description"],
                "icon": info["icon"],
                "color": info["color"],
            }

        # Find dominant dimension
        dominant_name = max(scores, key=scores.get)
        dominant = dimensions[dominant_name]

        # Generate summary
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        summary_parts = []
        for name, score in top_3:
            info = EMOTIONAL_DIMENSIONS[name]
            if score > 0.6:
                summary_parts.append(f"strong {info['label'].lower()}")
            elif score > 0.4:
                summary_parts.append(f"moderate {info['label'].lower()}")

        if summary_parts:
            summary = (
                f"This article primarily activates {summary_parts[0]} processing"
            )
            if len(summary_parts) > 1:
                summary += f", with {summary_parts[1]}"
            if len(summary_parts) > 2:
                summary += f" and {summary_parts[2]}"
            summary += "."
        else:
            summary = "This article produces a relatively uniform brain activation pattern."

        return {
            "dimensions": dimensions,
            "dominant": dominant_name,
            "summary": summary,
            "mode": "tribev2" if self.model is not None else "heuristic",
        }


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuroReader API",
    description="Predict emotional brain activation from article text using TRIBE v2",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chrome extension needs this
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize model on startup
tribe: Optional[TribeWrapper] = None


@app.on_event("startup")
async def startup():
    global tribe
    logging.basicConfig(level=logging.INFO)
    tribe = TribeWrapper()


class AnalyzeRequest(BaseModel):
    text: str
    url: Optional[str] = None
    title: Optional[str] = None


class DimensionScore(BaseModel):
    score: float
    label: str
    description: str
    icon: str
    color: str


class AnalyzeResponse(BaseModel):
    dimensions: dict[str, DimensionScore]
    dominant: str
    summary: str
    mode: str  # "tribev2" or "heuristic"


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(req: AnalyzeRequest):
    """Analyze article text and return emotional brain activation profile."""
    if not req.text or len(req.text.strip()) < 50:
        raise HTTPException(400, "Article text must be at least 50 characters.")

    # Truncate to ~5000 words to keep inference reasonable
    words = req.text.split()
    if len(words) > 5000:
        text = " ".join(words[:5000])
    else:
        text = req.text

    result = tribe.predict_from_text(text)
    return result


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": tribe is not None and tribe.model is not None,
        "mode": "tribev2" if (tribe and tribe.model) else "heuristic",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8421)
