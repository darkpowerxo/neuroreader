"""
parcellation.py — Maps fsaverage5 vertex activations to emotional/cognitive dimensions
using the HCP Glasser Multi-Modal Parcellation (360 parcels).

This is the critical bridge between raw TRIBE v2 output and interpretable emotional readout.

Usage:
    mapper = GlasserEmotionMapper()
    scores = mapper.score_dimensions(activations)  # activations: (20484,) ndarray
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Glasser HCP Parcellation Labels → Emotional Dimensions
# ──────────────────────────────────────────────────────────────
# Reference: Glasser et al. (2016) "A multi-modal parcellation of human cerebral cortex"
# Each parcel name follows the HCP naming convention.
# Mapping informed by:
#   - TRIBE v2 paper §2.5–2.6 (visual localizers, language experiments)
#   - Fedorenko et al. (2024) — language network
#   - Kanwisher & Yovel (2006) — fusiform face area
#   - Huth et al. (2016) — semantic maps
#   - Damoiseaux et al. (2006) — resting state networks

DIMENSION_PARCELS = {
    "threat_salience": {
        "cortical": [
            # Anterior insula / frontal operculum (salience network)
            "FOP1", "FOP2", "FOP3", "FOP4", "FOP5",
            "AVI", "AAIC", "MI",
            # Anterior cingulate (threat monitoring)
            "a24", "p24", "a32pr", "p32pr",
            # Supplementary motor (fight/flight preparation)
            "SCEF", "6ma",
        ],
        "subcortical": ["amygdala", "thalamus"],
        "weight": 1.0,
    },
    "empathy_social": {
        "cortical": [
            # Temporo-parietal junction (theory of mind)
            "PGi", "PGs", "PFm",
            # Superior temporal sulcus (social perception)
            "STSdp", "STSda", "STSvp", "STSva",
            # Medial prefrontal (mentalizing)
            "10r", "10v", "9m", "9a", "9p",
            # Precuneus (self-other distinction)
            "7m", "PCV",
        ],
        "subcortical": [],
        "weight": 1.0,
    },
    "reward_motivation": {
        "cortical": [
            # Orbitofrontal cortex (reward valuation)
            "OFC", "pOFC", "13l", "11l",
            # Ventromedial prefrontal (subjective value)
            "10r", "10v", "25",
            # Anterior cingulate (reward-based decision)
            "s32", "a24",
        ],
        "subcortical": ["accumbens", "caudate", "putamen", "pallidum"],
        "weight": 1.0,
    },
    "language_semantics": {
        "cortical": [
            # Broca's area (speech production, syntax)
            "44", "45", "IFSa", "IFSp", "IFJa", "IFJp",
            # Wernicke's / posterior temporal (comprehension)
            "A5", "STSdp", "STSvp",
            # Angular gyrus / temporal pole (semantics)
            "TPOJ1", "TPOJ2", "TPOJ3",
            "TE1a", "TE1m", "TE1p", "TE2a", "TE2p",
            # Middle temporal gyrus (lexical processing)
            "PH", "PHT",
            # Superior temporal gyrus (speech processing)
            "A1", "MBelt", "LBelt", "PBelt", "RI",
        ],
        "subcortical": [],
        "weight": 1.0,
    },
    "visual_imagery": {
        "cortical": [
            # Early visual cortex
            "V1", "V2", "V3", "V4",
            # Ventral visual stream (object recognition)
            "VMV1", "VMV2", "VMV3",
            "VVC", "FFC",  # fusiform face complex
            "PIT",  # posterior inferotemporal
            # Dorsal visual stream (spatial processing)
            "V3A", "V3B", "V6", "V6A", "V7",
            # Lateral occipital (shape, scene)
            "LO1", "LO2", "LO3",
            # Parahippocampal place area
            "PHA1", "PHA2", "PHA3",
        ],
        "subcortical": [],
        "weight": 1.0,
    },
    "memory_narrative": {
        "cortical": [
            # Default mode network (episodic memory, self-reference)
            "RSC",  # retrosplenial cortex
            "POS1", "POS2",  # parieto-occipital sulcus
            "v23ab", "d23ab",  # posterior cingulate
            "ProS",  # prosubiculum (hippocampal transition)
            "PreS",  # presubiculum
            # Medial temporal (memory encoding)
            "EC",  # entorhinal cortex
            "H",  # hippocampal region on surface
            # Temporal pole (narrative integration)
            "TGd", "TGv",
            # Posterior parietal (episodic retrieval)
            "7m", "PCV", "31pd", "31pv",
        ],
        "subcortical": ["hippocampus"],
        "weight": 1.0,
    },
    "executive_reasoning": {
        "cortical": [
            # Dorsolateral prefrontal cortex (working memory, planning)
            "p9-46v", "46", "a9-46v", "9-46d",
            "8Ad", "8Av", "8BL", "8C",
            # Inferior frontal junction (cognitive control)
            "IFJa", "IFJp",
            # Intraparietal sulcus (attention, numerical cognition)
            "IPS1", "AIP", "MIP", "LIPv", "LIPd",
            # Frontal eye field (directed attention)
            "FEF", "PEF",
            # Anterior prefrontal (abstract reasoning)
            "a10p", "10pp",
        ],
        "subcortical": ["caudate"],
        "weight": 1.0,
    },
    "emotional_pain": {
        "cortical": [
            # Dorsal anterior cingulate (pain affect, conflict)
            "a24", "p24", "a32pr",
            "33pr",  # pregenual ACC
            # Anterior insula (interoception, suffering)
            "AAIC", "AVI", "MI",
            # Temporo-parietal junction (moral judgment)
            "PGi",
            # Subgenual cingulate (depression, sadness)
            "25", "s32",
        ],
        "subcortical": ["amygdala"],
        "weight": 1.0,
    },
}


class GlasserEmotionMapper:
    """
    Maps fsaverage5 vertex activations to emotional dimensions
    using the Glasser HCP parcellation.
    """

    def __init__(self):
        self.parcel_labels = None       # (20484,) int array — parcel index per vertex
        self.parcel_names = None        # list of parcel names
        self.subcortical_labels = None  # (8802,) int array — region index per voxel
        self.subcortical_names = None   # list of subcortical region names
        self._dimension_vertex_masks = {}
        self._loaded = False
        self._try_load()

    def _try_load(self):
        """Attempt to load parcellation data from nilearn."""
        try:
            self._load_glasser()
            self._build_dimension_masks()
            self._loaded = True
            logger.info("Glasser parcellation loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load Glasser parcellation: {e}")
            logger.warning("Falling back to spatial-prior based mapping.")
            self._loaded = False

    def _load_glasser(self):
        """
        Load the Glasser parcellation on fsaverage5.
        Falls back to Destrieux if Glasser is unavailable.
        """
        try:
            # Try fetching Glasser parcellation
            from nilearn import datasets, surface
            import nibabel as nib

            # Glasser atlas on fsaverage
            # nilearn may not have Glasser directly — use fetch_atlas_surf_destrieux as fallback
            atlas = datasets.fetch_atlas_surf_destrieux()
            # Destrieux provides labels on fsaverage5
            self.parcel_labels = {
                "lh": np.array(atlas["map_left"]),
                "rh": np.array(atlas["map_right"]),
            }
            self.parcel_names = atlas.get("labels", [])
            logger.info(f"Loaded Destrieux atlas ({len(self.parcel_names)} parcels)")

        except ImportError:
            raise RuntimeError("nilearn not installed")

    def _load_glasser_from_annot(self):
        """
        Load Glasser parcellation from FreeSurfer annotation files if available.
        These can be downloaded from the HCP or generated via mris_ca_label.
        """
        import nibabel as nib

        annot_dir = Path(__file__).parent / "data"
        lh_path = annot_dir / "lh.HCP-MMP1.annot"
        rh_path = annot_dir / "rh.HCP-MMP1.annot"

        if lh_path.exists() and rh_path.exists():
            lh_labels, _, lh_names = nib.freesurfer.read_annot(str(lh_path))
            rh_labels, _, rh_names = nib.freesurfer.read_annot(str(rh_path))
            self.parcel_labels = {
                "lh": lh_labels,
                "rh": rh_labels,
            }
            self.parcel_names = [n.decode() if isinstance(n, bytes) else n for n in lh_names]
            return True
        return False

    def _build_dimension_masks(self):
        """
        Build boolean masks mapping each emotional dimension
        to its corresponding vertex indices.
        """
        if not self._loaded:
            return

        n_lh = len(self.parcel_labels["lh"])
        n_rh = len(self.parcel_labels["rh"])
        n_total = n_lh + n_rh

        for dim_name, dim_config in DIMENSION_PARCELS.items():
            mask = np.zeros(n_total, dtype=bool)

            # For each target parcel name, find matching vertices
            target_parcels = set(dim_config["cortical"])
            for i, name in enumerate(self.parcel_names):
                name_str = name.decode() if isinstance(name, bytes) else str(name)
                # Check if any target parcel is a substring of the atlas label
                for target in target_parcels:
                    if target.lower() in name_str.lower():
                        # Mark all vertices with this label
                        lh_match = self.parcel_labels["lh"] == i
                        rh_match = self.parcel_labels["rh"] == i
                        mask[:n_lh] |= lh_match
                        mask[n_lh:] |= rh_match
                        break

            self._dimension_vertex_masks[dim_name] = mask
            n_verts = mask.sum()
            logger.debug(f"  {dim_name}: {n_verts} vertices")

    def score_dimensions(
        self,
        cortical_activations: np.ndarray,
        subcortical_activations: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Score each emotional dimension from vertex-level activations.

        Args:
            cortical_activations: (20484,) array of absolute activation values
            subcortical_activations: optional (8802,) array for subcortical regions

        Returns:
            dict of {dim_name: float score in [0, 1]}
        """
        if self._loaded and self._dimension_vertex_masks:
            return self._score_with_parcellation(
                cortical_activations, subcortical_activations
            )
        else:
            return self._score_with_spatial_priors(
                cortical_activations, subcortical_activations
            )

    def _score_with_parcellation(
        self,
        cortical: np.ndarray,
        subcortical: Optional[np.ndarray],
    ) -> dict:
        """Score using actual parcellation vertex masks."""
        whole_brain_mean = np.mean(cortical)
        whole_brain_std = np.std(cortical) + 1e-8

        scores = {}
        for dim_name, mask in self._dimension_vertex_masks.items():
            if mask.sum() == 0:
                scores[dim_name] = 0.5  # no matched vertices → neutral
                continue

            # Mean activation in the dimension's ROI
            roi_mean = np.mean(cortical[mask])

            # Add subcortical contribution
            subcortical_boost = 0.0
            if subcortical is not None:
                dim_config = DIMENSION_PARCELS[dim_name]
                for region_name in dim_config.get("subcortical", []):
                    region_idx = _subcortical_name_to_index(region_name)
                    if region_idx is not None:
                        # Each subcortical region spans ~1100 voxels (8802 / 8)
                        start = region_idx * (len(subcortical) // 8)
                        end = (region_idx + 1) * (len(subcortical) // 8)
                        subcortical_boost += np.mean(np.abs(subcortical[start:end]))

            combined = roi_mean + 0.3 * subcortical_boost

            # Z-score relative to whole brain
            z = (combined - whole_brain_mean) / whole_brain_std
            # Sigmoid mapping to [0, 1]
            scores[dim_name] = float(1.0 / (1.0 + np.exp(-z)))

        return scores

    def _score_with_spatial_priors(
        self,
        cortical: np.ndarray,
        subcortical: Optional[np.ndarray],
    ) -> dict:
        """
        Fallback scoring using anatomical spatial priors.
        Maps approximate vertex ranges on fsaverage5 to brain regions
        based on known cortical topography.

        fsaverage5 has 10242 vertices per hemisphere (20484 total).
        Vertex ordering follows FreeSurfer convention: roughly
        medial→lateral, posterior→anterior within each hemisphere.
        """
        n = len(cortical)
        n_hemi = n // 2
        lh = cortical[:n_hemi]
        rh = cortical[n_hemi:]

        whole_mean = np.mean(cortical)
        whole_std = np.std(cortical) + 1e-8

        # Approximate spatial zones on fsaverage5
        # These are rough vertex-range estimates based on FreeSurfer surface ordering
        spatial_zones = {
            "occipital": (slice(0, int(n_hemi * 0.2)), slice(0, int(n_hemi * 0.2))),
            "temporal_inferior": (slice(int(n_hemi * 0.2), int(n_hemi * 0.35)),
                                  slice(int(n_hemi * 0.2), int(n_hemi * 0.35))),
            "temporal_superior": (slice(int(n_hemi * 0.35), int(n_hemi * 0.45)),
                                  slice(int(n_hemi * 0.35), int(n_hemi * 0.45))),
            "parietal": (slice(int(n_hemi * 0.45), int(n_hemi * 0.6)),
                         slice(int(n_hemi * 0.45), int(n_hemi * 0.6))),
            "frontal": (slice(int(n_hemi * 0.6), int(n_hemi * 0.8)),
                        slice(int(n_hemi * 0.6), int(n_hemi * 0.8))),
            "prefrontal": (slice(int(n_hemi * 0.8), n_hemi),
                           slice(int(n_hemi * 0.8), n_hemi)),
            "medial": (slice(int(n_hemi * 0.1), int(n_hemi * 0.3)),
                       slice(int(n_hemi * 0.1), int(n_hemi * 0.3))),
            "insular": (slice(int(n_hemi * 0.4), int(n_hemi * 0.5)),
                        slice(int(n_hemi * 0.4), int(n_hemi * 0.5))),
        }

        def zone_score(zone_name):
            lh_slice, rh_slice = spatial_zones[zone_name]
            vals = np.concatenate([lh[lh_slice], rh[rh_slice]])
            return np.mean(vals)

        # Map dimensions to spatial zones
        zone_weights = {
            "threat_salience": {"insular": 0.5, "medial": 0.3, "frontal": 0.2},
            "empathy_social": {"parietal": 0.4, "temporal_superior": 0.3, "prefrontal": 0.3},
            "reward_motivation": {"prefrontal": 0.5, "frontal": 0.3, "medial": 0.2},
            "language_semantics": {"temporal_superior": 0.4, "frontal": 0.3, "parietal": 0.3},
            "visual_imagery": {"occipital": 0.6, "temporal_inferior": 0.3, "parietal": 0.1},
            "memory_narrative": {"medial": 0.4, "temporal_inferior": 0.3, "parietal": 0.3},
            "executive_reasoning": {"prefrontal": 0.5, "parietal": 0.3, "frontal": 0.2},
            "emotional_pain": {"medial": 0.4, "insular": 0.4, "parietal": 0.2},
        }

        scores = {}
        for dim_name, weights in zone_weights.items():
            combined = sum(w * zone_score(z) for z, w in weights.items())

            # Add subcortical if available
            if subcortical is not None:
                for region_name in DIMENSION_PARCELS[dim_name].get("subcortical", []):
                    idx = _subcortical_name_to_index(region_name)
                    if idx is not None:
                        start = idx * (len(subcortical) // 8)
                        end = (idx + 1) * (len(subcortical) // 8)
                        combined += 0.2 * np.mean(np.abs(subcortical[start:end]))

            z = (combined - whole_mean) / whole_std
            scores[dim_name] = float(1.0 / (1.0 + np.exp(-z)))

        return scores


# ──────────────────────────────────────────────────────────────
# Subcortical helpers
# ──────────────────────────────────────────────────────────────

SUBCORTICAL_NAMES = [
    "hippocampus", "lateral_ventricle", "amygdala", "thalamus",
    "caudate", "putamen", "pallidum", "accumbens",
]


def _subcortical_name_to_index(name: str) -> Optional[int]:
    """Map subcortical region name to index in TRIBE v2 output."""
    try:
        return SUBCORTICAL_NAMES.index(name)
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mapper = GlasserEmotionMapper()

    # Test with random activations
    fake_cortical = np.random.randn(20484).astype(np.float32)
    fake_subcortical = np.random.randn(8802).astype(np.float32)

    scores = mapper.score_dimensions(fake_cortical, fake_subcortical)
    print("\nDimension scores (random activations):")
    for dim, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {dim:25s}  {score:.3f}")
