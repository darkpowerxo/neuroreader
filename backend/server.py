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
# Image utilities
# ---------------------------------------------------------------------------

def download_images(urls: list[str], max_images: int = 10, timeout: int = 10) -> list[str]:
    """Download images from URLs to temp files. Returns list of local file paths."""
    import urllib.request
    from urllib.parse import urlparse

    paths = []
    for url in urls[:max_images]:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                continue
            suffix = Path(parsed.path).suffix or ".jpg"
            if suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
                suffix = ".jpg"
            req = urllib.request.Request(url, headers={"User-Agent": "NeuroReader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read(10 * 1024 * 1024)  # 10MB max per image
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            paths.append(tmp.name)
        except Exception as e:
            logging.warning(f"Failed to download image {url}: {e}")
    return paths


def images_to_video(image_paths: list[str], fps: int = 1, duration_per_image: float = 2.0) -> str | None:
    """Stitch images into a short video for TRIBE v2 video pathway.
    Each image is shown for `duration_per_image` seconds."""
    if not image_paths:
        return None
    try:
        from moviepy import ImageClip, concatenate_videoclips

        clips = []
        for img_path in image_paths:
            try:
                clip = ImageClip(img_path).with_duration(duration_per_image)
                clips.append(clip)
            except Exception as e:
                logging.warning(f"Skipping image {img_path}: {e}")

        if not clips:
            return None

        video = concatenate_videoclips(clips, method="compose")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        video.write_videofile(tmp.name, fps=fps, logger=None, audio=False)
        video.close()
        for c in clips:
            c.close()
        return tmp.name
    except Exception as e:
        logging.warning(f"Failed to create video from images: {e}")
        return None


def analyze_images_heuristic(image_paths: list[str]) -> dict:
    """Basic image analysis for heuristic mode — color/brightness signals."""
    from PIL import Image

    results = {
        "count": len(image_paths),
        "avg_brightness": 0.5,
        "red_dominance": 0.0,
        "blue_dominance": 0.0,
        "dark_ratio": 0.0,
        "has_faces": False,
    }

    if not image_paths:
        return results

    brightnesses = []
    red_scores = []
    blue_scores = []
    dark_counts = 0

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB").resize((128, 128))
            pixels = np.array(img, dtype=np.float32) / 255.0
            r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]

            brightness = pixels.mean()
            brightnesses.append(brightness)

            # Color channel dominance
            red_mask = (r > g + 0.1) & (r > b + 0.1)
            blue_mask = (b > r + 0.1) & (b > g + 0.1)
            red_scores.append(float(red_mask.mean()))
            blue_scores.append(float(blue_mask.mean()))

            if brightness < 0.35:
                dark_counts += 1
        except Exception:
            continue

    if brightnesses:
        results["avg_brightness"] = float(np.mean(brightnesses))
        results["red_dominance"] = float(np.mean(red_scores))
        results["blue_dominance"] = float(np.mean(blue_scores))
        results["dark_ratio"] = dark_counts / len(image_paths)

    return results
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
            import pathlib
            import platform

            # Fix Windows PosixPath unpickling: checkpoints saved on Linux
            # contain pickled PosixPath objects that can't be instantiated on
            # Windows. Monkey-patch so they deserialize as WindowsPath instead.
            if platform.system() == "Windows":
                pathlib.PosixPath = pathlib.WindowsPath

            from tribev2 import TribeModel

            # Fix Windows Path mangling the HuggingFace repo id
            # ("facebook/tribev2" → "facebook\tribev2").
            import huggingface_hub as _hf
            _orig_download = _hf.hf_hub_download

            def _fixed_download(repo_id, *args, **kwargs):
                # On Windows, Path("facebook/tribev2") → "facebook\\tribev2"
                repo_id = repo_id.replace("\\", "/")
                return _orig_download(repo_id, *args, **kwargs)

            _hf.hf_hub_download = _fixed_download

            # Fix whisperx invocation: TRIBE v2 calls `uvx whisperx` which
            # creates an isolated env with CPU-only torch. Monkey-patch to
            # use our venv's whisperx binary (CUDA-enabled) instead.
            import shutil
            import sys
            from tribev2 import eventstransforms as _et
            _orig_get_transcript = _et.ExtractWordsFromAudio._get_transcript_from_audio

            def _patched_get_transcript(self_wx, wav_filename, language):
                """Replace 'uvx whisperx' with our venv's whisperx."""
                import subprocess, tempfile, json
                import pandas as pd

                language_codes = dict(
                    english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
                )
                if language not in language_codes:
                    raise ValueError(f"Language {language} not supported")

                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "float32"

                # Use our venv's whisperx, not uvx
                venv_whisperx = shutil.which("whisperx")
                if not venv_whisperx:
                    venv_whisperx = str(Path(sys.executable).parent / "whisperx.exe")

                with tempfile.TemporaryDirectory() as output_dir:
                    logging.info(f"Running whisperx from venv on {device}...")
                    cmd = [
                        venv_whisperx,
                        str(wav_filename),
                        "--model", "large-v3",
                        "--language", language_codes[language],
                        "--device", device,
                        "--compute_type", compute_type,
                        "--batch_size", "16",
                        "--output_dir", output_dir,
                        "--output_format", "json",
                    ]
                    if language == "english":
                        cmd += ["--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"]
                    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
                    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    if result.returncode != 0:
                        raise RuntimeError(f"whisperx failed:\n{result.stderr}")

                    json_path = Path(output_dir) / f"{Path(wav_filename).stem}.json"
                    transcript = json.loads(json_path.read_text())

                words = []
                for i, segment in enumerate(transcript["segments"]):
                    sentence = segment["text"].replace('"', "")
                    for word in segment["words"]:
                        if "start" not in word:
                            continue
                        words.append({
                            "text": word["word"].replace('"', ""),
                            "start": word["start"],
                            "duration": word["end"] - word["start"],
                            "sequence_id": i,
                            "sentence": sentence,
                        })
                return pd.DataFrame(words)

            _et.ExtractWordsFromAudio._get_transcript_from_audio = _patched_get_transcript

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

    def predict_from_text(self, text: str, images: list[str] | None = None,
                          enable_audio: bool = True, enable_video: bool = True) -> dict:
        """
        Given article text, predict brain activations and return emotional profile.

        Args:
            text: The article text to analyze.
            images: Optional list of image URLs from the article.
            enable_audio: Whether to use TTS + WhisperX audio pathway.
            enable_video: Whether to stitch images into video for visual pathway.

        Returns dict with:
            - dimensions: {dim_name: {score, label, description, ...}}
            - dominant: the highest-scoring dimension
            - summary: human-readable interpretation
        """
        if self.model is not None:
            try:
                return self._predict_real(text, images=images,
                                          enable_audio=enable_audio, enable_video=enable_video)
            except OSError as e:
                if "gated repo" in str(e).lower():
                    logging.error(
                        "LLaMA 3.2-3B access denied — your HuggingFace request is still pending. "
                        "Visit https://huggingface.co/meta-llama/Llama-3.2-3B to check status. "
                        "Falling back to heuristic mode."
                    )
                else:
                    logging.error(f"TRIBE v2 inference failed: {e}. Falling back to heuristic mode.")
                return self._predict_mock(text, images=images)
            except Exception as e:
                logging.error(f"TRIBE v2 inference failed: {e}. Falling back to heuristic mode.")
                return self._predict_mock(text, images=images)
        else:
            return self._predict_mock(text, images=images)

    def _create_text_events_without_audio(self, text: str) -> "pd.DataFrame":
        """Create word-level events directly from text, bypassing TTS/WhisperX.

        Produces the same DataFrame structure as the TTS→WhisperX pipeline
        but with synthetic timestamps, then applies the standard text transforms
        (AddText, AddSentenceToWords, AddContextToWords, RemoveMissing).
        """
        import re
        import pandas as pd
        from neuralset.events.utils import standardize_events
        from neuralset.events.transforms.text import (
            AddText, AddSentenceToWords, AddContextToWords,
        )
        from neuralset.events.transforms.basic import RemoveMissing
        from langdetect import detect

        lang_code = detect(text)
        # Map 2-letter code to full name for TRIBE v2
        lang_map = {"en": "english", "fr": "french", "es": "spanish",
                     "nl": "dutch", "zh": "chinese"}
        language = lang_map.get(lang_code, "english")

        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [text.strip()]

        # Create Word events with synthetic timestamps (~200 WPM)
        words = []
        t = 0.0
        word_duration = 0.3
        for seq_id, sentence in enumerate(sentences):
            for word_text in sentence.split():
                if not word_text:
                    continue
                words.append({
                    "type": "Word",
                    "text": word_text,
                    "start": t,
                    "duration": word_duration,
                    "sequence_id": seq_id,
                    "sentence": sentence,
                    "language": language,
                    "timeline": "default",
                    "subject": "default",
                })
                t += word_duration

        if not words:
            raise ValueError("No words could be extracted from text")

        df = pd.DataFrame(words)
        df = standardize_events(df)

        transforms = [
            AddText(),
            AddSentenceToWords(max_unmatched_ratio=0.05),
            AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
            RemoveMissing(),
        ]
        for transform in transforms:
            df = transform(df)

        return standardize_events(df, auto_fill=False)

    def _predict_real(self, text: str, images: list[str] | None = None,
                      enable_audio: bool = True, enable_video: bool = True) -> dict:
        """Run actual TRIBE v2 inference, optionally with images via video pathway."""
        text_path = None
        video_path = None
        image_paths = []
        try:
            # If images are provided and video is enabled, download and stitch into a video
            if images and enable_video:
                image_paths = download_images(images)
                if image_paths:
                    video_path = images_to_video(image_paths)
                    if video_path:
                        logging.info(f"Created video from {len(image_paths)} images for TRIBE v2 visual pathway")

            if enable_audio:
                # Full pipeline: text → TTS → WhisperX → word events
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                    f.write(text)
                    text_path = f.name
                if video_path:
                    df = self.model.get_events_dataframe(video_path=video_path)
                else:
                    df = self.model.get_events_dataframe(text_path=text_path)
            else:
                # Fast path: create word events directly from text, skip TTS/WhisperX
                logging.info("Audio disabled — creating word events directly from text")
                df = self._create_text_events_without_audio(text)

            # Predict: returns (n_timesteps, n_vertices) cortical activations
            preds, segments = self.model.predict(events=df)

            # Average across time to get a spatial activation map
            mean_activation = np.mean(np.abs(preds), axis=0)  # (n_vertices,)

            # Map to emotional dimensions
            return self._activations_to_emotions(mean_activation)
        finally:
            if text_path:
                try:
                    os.unlink(text_path)
                except OSError:
                    pass
            if video_path:
                try:
                    os.unlink(video_path)
                except OSError:
                    pass
            for p in image_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

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

    def _predict_mock(self, text: str, images: list[str] | None = None) -> dict:
        """
        Mock prediction using heuristic NLP analysis + image analysis.
        Used when TRIBE v2 is not installed — gives a demo experience.
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Download and analyze images if provided
        image_analysis = {"count": 0, "avg_brightness": 0.5, "red_dominance": 0.0,
                          "blue_dominance": 0.0, "dark_ratio": 0.0}
        image_paths = []
        if images:
            try:
                image_paths = download_images(images)
                if image_paths:
                    image_analysis = analyze_images_heuristic(image_paths)
                    logging.info(f"Analyzed {len(image_paths)} images: brightness={image_analysis['avg_brightness']:.2f}, "
                                 f"red={image_analysis['red_dominance']:.2f}, dark={image_analysis['dark_ratio']:.2f}")
            except Exception as e:
                logging.warning(f"Image analysis failed: {e}")
            finally:
                for p in image_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

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

        # Boost scores based on actual image analysis
        n_images = image_analysis["count"]
        if n_images > 0:
            # Visual imagery: boosted by image presence
            img_boost = min(n_images * 0.1, 0.35)
            scores["visual_imagery"] = min(scores["visual_imagery"] + img_boost, 1.0)

            # Threat: dark images and red-dominant images signal danger
            dark_signal = image_analysis["dark_ratio"] * 0.25
            red_signal = image_analysis["red_dominance"] * 0.2
            scores["threat_salience"] = min(scores["threat_salience"] + dark_signal + red_signal, 1.0)

            # Emotional pain: dark imagery correlates with distress
            scores["emotional_pain"] = min(scores["emotional_pain"] + dark_signal * 0.8, 1.0)

            # Empathy: presence of many images suggests human stories
            if n_images >= 3:
                scores["empathy_social"] = min(scores["empathy_social"] + 0.1, 1.0)

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
    images: Optional[list[str]] = None
    enable_audio: bool = True
    enable_video: bool = True


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

    result = tribe.predict_from_text(
        text,
        images=req.images,
        enable_audio=req.enable_audio,
        enable_video=req.enable_video,
    )
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
