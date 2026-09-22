import json
import os
import logging
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_URL = os.getenv(
    "GROQ_URL",
    "https://api.groq.com/openai/v1/chat/completions",
).rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


class SupervisorDecision(BaseModel):
    classes: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    workflow: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    router: str = Field(default="keyword")  # "rl" or "keyword"


CLASS_DESCRIPTIONS = {
    "SINGLE_IMAGE": "Single-image VQA, scene understanding, or counting.",
    "GROUNDING_CAPTIONING": "Single-image object location and grounding.",
    "CHANGE_ANALYSIS": "Two corresponding images from different dates.",
    "OPTICAL_SAR": "Two complementary optical and SAR observations.",
    "GENERAL_REMOTE_SENSING": "Other remote-sensing analysis.",
}
VALID_CLASSES = set(CLASS_DESCRIPTIONS)

SUPERVISOR_SYSTEM_PROMPT = """
Classify a remote-sensing request. Return JSON only with classes, workflow,
and parameters. Use GROUNDING_CAPTIONING for object location questions,
CHANGE_ANALYSIS for two-date change questions, OPTICAL_SAR for any other
two-image request, and SINGLE_IMAGE for single-image VQA or counting.
"""


def route_query(query: str, image_count: int) -> SupervisorDecision:
    normalized = query.lower().strip()
    logger.info("routing_started image_count=%s query=%r", image_count, query)
    change_terms = (
        "change", "changed", "difference", "before and after",
        "between dates", "increase", "decrease", "removed", "added",
        "construction",
    )
    count_terms = ("how many", "number of", "count", "quantity", "total")
    grounding_terms = (
        "where", "locate", "location", "highlight", "point out", "identify",
        "find", "which region", "which area",
    )
    spatial_terms = (
        "top", "bottom", "left", "right", "upper", "lower", "corner",
        "center", "centre", "near", "alongside", "beside",
    )
    visual_target_terms = (
        "white", "black", "red", "blue", "green", "yellow", "bright",
        "dark", "large", "small",
    )
    vqa_question_terms = (
        "what is", "what's", "what are", "describe", "explain", "tell me",
        "classify", "identify the type", "what kind",
    )
    description_terms = (
        "describe", "description", "caption", "summarize", "summary",
        "what do you see", "what is in the image", "scene",
    )
    object_terms = (
        "vehicle", "vehicles", "car", "cars", "truck", "trucks", "ship",
        "ships", "boat", "boats", "aircraft", "plane", "planes", "building",
        "buildings", "road", "roads", "runway", "tank", "tanks", "object",
        "objects", "thing", "things", "house", "houses", "home", "homes",
    )

    if image_count >= 2:
        if any(term in normalized for term in change_terms):
            decision = SupervisorDecision(
                classes=["CHANGE_ANALYSIS"],
                workflow=["bi_temporal_change_analysis"],
            )
            logger.info("routing_decision classes=%s workflow=%s", decision.classes, decision.workflow)
            return decision
        decision = SupervisorDecision(
            classes=["OPTICAL_SAR"],
            workflow=["optical_sar_fusion"],
        )
        logger.info("routing_decision classes=%s workflow=%s", decision.classes, decision.workflow)
        return decision

    if any(term in normalized for term in count_terms):
        decision = SupervisorDecision(
            classes=["SINGLE_IMAGE"],
            workflow=["visual_question_answering"],
            parameters={"intent": "counting"},
        )
        logger.info("routing_decision classes=%s workflow=%s parameters=%s", decision.classes, decision.workflow, decision.parameters)
        return decision

    if any(term in normalized for term in description_terms):
        decision = SupervisorDecision(
            classes=["SINGLE_IMAGE", "GROUNDING_CAPTIONING"],
            workflow=["visual_question_answering", "scene_captioning"],
            parameters={"intent": "description", "combine_outputs": True},
        )
        logger.info("routing_decision classes=%s workflow=%s parameters=%s", decision.classes, decision.workflow, decision.parameters)
        return decision

    is_explicit_grounding = any(term in normalized for term in grounding_terms)
    is_explicit_vqa = any(term in normalized for term in vqa_question_terms) and not is_explicit_grounding
    is_grounding_request = (is_explicit_grounding or any(
        term in normalized for term in spatial_terms
    )) and not is_explicit_vqa
    has_target = any(term in normalized for term in object_terms) or any(
        term in normalized for term in visual_target_terms
    )
    if is_grounding_request and has_target:
        decision = SupervisorDecision(
            classes=["GROUNDING_CAPTIONING"],
            workflow=["object_grounding_and_captioning"],
            parameters={"intent": "grounding"},
        )
        logger.info("routing_decision classes=%s workflow=%s parameters=%s", decision.classes, decision.workflow, decision.parameters)
        return decision

    decision = SupervisorDecision(
        classes=["SINGLE_IMAGE"],
        workflow=["visual_question_answering"],
    )
    logger.info("routing_decision classes=%s workflow=%s", decision.classes, decision.workflow)
    return decision


def call_groq_supervisor(
    query: str,
    input_info: Dict[str, Any],
) -> SupervisorDecision:
    routed = route_query(query, int(input_info.get("image_count", 0)))
    # The local rules encode the product contract and must not be overridden
    # by a probabilistic supervisor response.
    logger.info("routing_authoritative classes=%s workflow=%s", routed.classes, routed.workflow)
    return routed
