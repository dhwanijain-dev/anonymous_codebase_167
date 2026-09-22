"""
rl_router.py — RL-based A2A router for SatQuery

Wraps SatQueryRouter (DQN trained in satquery_router_rl.py) and produces
a SupervisorDecision compatible with the existing vlm.execute_workflow()
contract so that main.py can swap it in for call_groq_supervisor() with
zero changes to the downstream pipeline.

Checkpoint path is resolved from:
  1. Environment variable  ROUTER_CHECKPOINT
  2. Default: backend/satquery_router.pt  (sibling of this file)

If no checkpoint exists yet, the module falls back to the original
keyword-based router from supervisor.py and logs a warning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RL-action → vlm.py class label mapping
# ---------------------------------------------------------------------------
# The RL model uses its own action names; vlm.execute_workflow() expects the
# legacy class names that drive WORKFLOW_TO_ADAPTER / WORKFLOW_TO_REGISTRY.

RL_ACTION_TO_VLM_CLASS: Dict[str, str] = {
    "SINGLE_IMAGE_VQA":     "SINGLE_IMAGE",
    "GROUNDING_CAPTIONING": "GROUNDING_CAPTIONING",
    "CHANGE_ANALYSIS":      "CHANGE_ANALYSIS",
    "OPTICAL_SAR_ANALYSIS": "OPTICAL_SAR",
    "GENERAL_AGENT":        "GENERAL_REMOTE_SENSING",
    "END":                  "",          # sentinel — filtered out below
}

# vlm.py workflow label (used in SupervisorDecision.workflow list)
RL_ACTION_TO_WORKFLOW: Dict[str, str] = {
    "SINGLE_IMAGE_VQA":     "visual_question_answering",
    "GROUNDING_CAPTIONING": "object_grounding_and_captioning",
    "CHANGE_ANALYSIS":      "bi_temporal_change_analysis",
    "OPTICAL_SAR_ANALYSIS": "optical_sar_fusion",
    "GENERAL_AGENT":        "general_remote_sensing",
}

# Grounding intent is inferred when GROUNDING_CAPTIONING is routed
GROUNDING_KEYWORDS = (
    "where", "locate", "location", "highlight", "point out",
    "find", "which region", "which area", "identify",
)


# ---------------------------------------------------------------------------
# Lazy singleton for the RL checkpoint
# ---------------------------------------------------------------------------

_router = None          # SatQueryRouter instance or None
_router_loaded = False  # True once we've attempted to load


def _load_router() -> None:
    """Attempt to load the RL checkpoint once; silently degrade if missing."""
    global _router, _router_loaded
    if _router_loaded:
        return
    _router_loaded = True

    checkpoint_path = os.getenv(
        "ROUTER_CHECKPOINT",
        str(Path(__file__).parent / "satquery_router.pt"),
    )

    if not Path(checkpoint_path).exists():
        logger.warning(
            "rl_router: checkpoint not found at %s — "
            "falling back to keyword router. "
            "Run satquery_router_rl.py to train and produce the checkpoint.",
            checkpoint_path,
        )
        return

    try:
        # Import here so the module is importable even when torch is absent.
        from satquery_router_rl import SatQueryRouter  # type: ignore

        _router = SatQueryRouter(checkpoint_path)
        logger.info("rl_router: loaded checkpoint from %s", checkpoint_path)
    except Exception as exc:
        logger.error(
            "rl_router: failed to load checkpoint %s: %s — "
            "falling back to keyword router.",
            checkpoint_path,
            exc,
            exc_info=True,
        )
        _router = None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def route_with_rl(
    query: str,
    image_paths: List[str],
    modalities: List[str],
    *,
    max_steps: int = 4,
) -> "SupervisorDecision":
    """
    Run the DQN router and return a SupervisorDecision.

    Parameters
    ----------
    query:
        The user's natural-language question.
    image_paths:
        Absolute paths to the uploaded images (1 or 2).
    modalities:
        Per-image modality strings, e.g. ["optical"] or ["optical","sar"].
    max_steps:
        Maximum routing steps (default 4, matches training config).

    Returns
    -------
    SupervisorDecision
        Compatible with the existing vlm.execute_workflow() contract.
    """
    from supervisor import SupervisorDecision, route_query  # local import avoids circular deps

    _load_router()

    # ------------------------------------------------------------------
    # Fast path: no checkpoint → use keyword heuristic router
    # ------------------------------------------------------------------
    if _router is None:
        logger.info("rl_router: using keyword fallback")
        return route_query(query, len(image_paths))

    # ------------------------------------------------------------------
    # RL path
    # ------------------------------------------------------------------
    try:
        trace = _router.route(
            query=query,
            image_paths=image_paths,
            modalities=modalities,
            max_steps=max_steps,
        )
    except Exception as exc:
        logger.error(
            "rl_router: inference error (%s) — falling back to keyword router",
            exc,
            exc_info=True,
        )
        from supervisor import route_query
        return route_query(query, len(image_paths))

    logger.info("rl_router: trace=%s", trace)

    # ------------------------------------------------------------------
    # Translate RL trace → SupervisorDecision
    # ------------------------------------------------------------------
    classes: List[str] = []
    workflow: List[str] = []
    parameters: Dict[str, Any] = {}

    for step in trace:
        action_name: str = step["action"]

        if action_name == "END":
            break

        vlm_class = RL_ACTION_TO_VLM_CLASS.get(action_name)
        if not vlm_class:
            continue

        if vlm_class not in classes:
            classes.append(vlm_class)

        wf = RL_ACTION_TO_WORKFLOW.get(action_name)
        if wf and wf not in workflow:
            workflow.append(wf)

    # Grounding intent detection (keeps downstream grounding_requested logic)
    if "GROUNDING_CAPTIONING" in classes:
        query_lower = query.lower()
        if any(kw in query_lower for kw in GROUNDING_KEYWORDS):
            parameters["intent"] = "grounding"
        else:
            parameters["intent"] = "captioning"

    # Combine outputs when both VQA and grounding/captioning are present
    if "SINGLE_IMAGE" in classes and "GROUNDING_CAPTIONING" in classes:
        parameters["combine_outputs"] = True

    # Fallback: if RL produced no classes (e.g. routed straight to END)
    if not classes:
        logger.warning(
            "rl_router: no specialist selected — defaulting to SINGLE_IMAGE"
        )
        classes = ["SINGLE_IMAGE"]
        workflow = ["visual_question_answering"]

    decision = SupervisorDecision(
        classes=classes,
        workflow=workflow,
        parameters=parameters,
        reasoning=f"RL router trace: {[s['action'] for s in trace]}",
        router="rl",
    )

    logger.info(
        "rl_router: decision classes=%s workflow=%s parameters=%s",
        decision.classes,
        decision.workflow,
        decision.parameters,
    )
    return decision
