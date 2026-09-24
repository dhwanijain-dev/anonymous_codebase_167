"""
clf_router.py — Deep-learning classifier router for SatQuery
=============================================================

Replaces the old rl_router.py. Loads the trained SatQueryClassifier
(from satquery_classifier.py) and translates its multi-label predictions
into a SupervisorDecision compatible with vlm.execute_workflow().

Checkpoint resolution order:
    1. Environment variable  CLASSIFIER_CHECKPOINT
    2. Default: backend/satquery_clf.pt  (sibling of this file)

If no checkpoint is present the module falls back transparently to the
keyword-based router from supervisor.py and logs a warning.

Environment variables
---------------------
    CLASSIFIER_CHECKPOINT   Path to the .pt file produced by satquery_classifier.py
    CLF_THRESHOLD           Override sigmoid threshold (float 0-1, default 0.5)
    SATQUERY_USE_CLF_ROUTER true/false  (default true)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classifier label → vlm.py class label
# ---------------------------------------------------------------------------
# satquery_classifier.py uses:
#   SINGLE_IMAGE_VQA, GROUNDING_CAPTIONING, CHANGE_ANALYSIS, OPTICAL_SAR_ANALYSIS
# vlm.execute_workflow() expects:
#   SINGLE_IMAGE, GROUNDING_CAPTIONING, CHANGE_ANALYSIS, OPTICAL_SAR

CLF_LABEL_TO_VLM_CLASS: Dict[str, str] = {
    "SINGLE_IMAGE_VQA":     "SINGLE_IMAGE",
    "GROUNDING_CAPTIONING": "GROUNDING_CAPTIONING",
    "CHANGE_ANALYSIS":      "CHANGE_ANALYSIS",
    "OPTICAL_SAR_ANALYSIS": "OPTICAL_SAR",
}

CLF_LABEL_TO_WORKFLOW: Dict[str, str] = {
    "SINGLE_IMAGE_VQA":     "visual_question_answering",
    "GROUNDING_CAPTIONING": "object_grounding_and_captioning",
    "CHANGE_ANALYSIS":      "bi_temporal_change_analysis",
    "OPTICAL_SAR_ANALYSIS": "optical_sar_fusion",
}

# Keywords that signal a grounding intent when GROUNDING_CAPTIONING is predicted
GROUNDING_KEYWORDS = (
    "where", "locate", "location", "highlight", "point out",
    "find", "which region", "which area", "identify",
)


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_clf = None           # SatQueryClassifier instance
_clf_loaded = False   # True once a load attempt has been made


def _load_classifier() -> None:
    global _clf, _clf_loaded
    if _clf_loaded:
        return
    _clf_loaded = True

    checkpoint_path = os.getenv(
        "CLASSIFIER_CHECKPOINT",
        str(Path(__file__).parent / "satquery_clf.pt"),
    )

    if not Path(checkpoint_path).exists():
        logger.warning(
            "clf_router: checkpoint not found at '%s'. "
            "Falling back to keyword router. "
            "Run: python satquery_classifier.py --manifest router_manifest_combined.jsonl",
            checkpoint_path,
        )
        return

    try:
        from satquery_classifier import SatQueryClassifier  # type: ignore

        threshold = float(os.getenv("CLF_THRESHOLD", "0.5"))
        _clf = SatQueryClassifier(checkpoint_path)
        # Allow env-var override of the threshold stored in checkpoint
        _clf.threshold = threshold
        logger.info(
            "clf_router: loaded checkpoint from '%s' (threshold=%.2f)",
            checkpoint_path,
            threshold,
        )
    except Exception as exc:
        logger.error(
            "clf_router: failed to load checkpoint '%s': %s — falling back to keyword router.",
            checkpoint_path,
            exc,
            exc_info=True,
        )
        _clf = None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def route_with_clf(
    query: str,
    image_paths: List[str],
    modalities: List[str],
    threshold: float | None = None,
) -> "SupervisorDecision":
    """
    Classify the request and return a SupervisorDecision.

    Parameters
    ----------
    query       : Natural-language question from the user.
    image_paths : Absolute paths to uploaded images (1 or 2).
    modalities  : Per-image modality strings, e.g. ["optical", "sar"].
    threshold   : Override sigmoid threshold for this call.

    Returns
    -------
    SupervisorDecision  compatible with vlm.execute_workflow().
    """
    from supervisor import SupervisorDecision, route_query  # local to avoid circular import

    _load_classifier()

    # ------------------------------------------------------------------
    # Fallback: no checkpoint → keyword routing
    # ------------------------------------------------------------------
    if _clf is None:
        logger.info("clf_router: no classifier loaded — using keyword fallback")
        decision = route_query(query, len(image_paths))
        decision.router = "keyword"
        return decision

    # ------------------------------------------------------------------
    # Classifier path
    # ------------------------------------------------------------------
    try:
        clf_labels = _clf.predict(
            query=query,
            image_paths=image_paths,
            modalities=modalities,
            threshold=threshold,
        )
        proba = _clf.predict_proba(
            query=query,
            image_paths=image_paths,
            modalities=modalities,
        )
    except Exception as exc:
        logger.error(
            "clf_router: inference error (%s) — falling back to keyword router",
            exc,
            exc_info=True,
        )
        decision = route_query(query, len(image_paths))
        decision.router = "keyword"
        return decision

    logger.info("clf_router: raw predictions=%s proba=%s", clf_labels, proba)

    # ------------------------------------------------------------------
    # Build SupervisorDecision from multi-label predictions
    # ------------------------------------------------------------------
    classes:  List[str] = []
    workflow: List[str] = []
    parameters: Dict[str, Any] = {"probabilities": proba}

    for clf_label in clf_labels:
        vlm_class = CLF_LABEL_TO_VLM_CLASS.get(clf_label)
        wf_label  = CLF_LABEL_TO_WORKFLOW.get(clf_label)
        if vlm_class and vlm_class not in classes:
            classes.append(vlm_class)
        if wf_label and wf_label not in workflow:
            workflow.append(wf_label)

    # Grounding intent
    if "GROUNDING_CAPTIONING" in classes:
        query_lower = query.lower()
        if any(kw in query_lower for kw in GROUNDING_KEYWORDS):
            parameters["intent"] = "grounding"
        else:
            parameters["intent"] = "captioning"

    # Combine outputs when both VQA + Grounding/Captioning are requested
    if "SINGLE_IMAGE" in classes and "GROUNDING_CAPTIONING" in classes:
        parameters["combine_outputs"] = True

    # Hard safety: multi-image tasks require 2 images
    if len(image_paths) < 2:
        classes  = [c for c in classes if c not in {"CHANGE_ANALYSIS", "OPTICAL_SAR"}]
        workflow = [w for w in workflow if w not in {"bi_temporal_change_analysis", "optical_sar_fusion"}]

    # Final fallback if everything was stripped
    if not classes:
        logger.warning("clf_router: no valid classes after filtering — defaulting to SINGLE_IMAGE")
        classes  = ["SINGLE_IMAGE"]
        workflow = ["visual_question_answering"]

    decision = SupervisorDecision(
        classes=classes,
        workflow=workflow,
        parameters=parameters,
        reasoning=(
            f"Classifier predictions: {clf_labels} | "
            f"proba: { {k: f'{v:.2f}' for k, v in proba.items()} }"
        ),
        router="classifier",
    )

    logger.info(
        "clf_router: decision classes=%s workflow=%s parameters=%s",
        decision.classes,
        decision.workflow,
        {k: v for k, v in parameters.items() if k != "probabilities"},
    )
    return decision
