import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel, Field

load_dotenv()

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
    object_terms = (
        "vehicle", "vehicles", "car", "cars", "truck", "trucks", "ship",
        "ships", "boat", "boats", "aircraft", "plane", "planes", "building",
        "buildings", "road", "roads", "runway", "tank", "tanks", "object",
        "objects", "thing", "things",
    )

    if image_count >= 2:
        if any(term in normalized for term in change_terms):
            return SupervisorDecision(
                classes=["CHANGE_ANALYSIS"],
                workflow=["bi_temporal_change_analysis"],
            )
        return SupervisorDecision(
            classes=["OPTICAL_SAR"],
            workflow=["optical_sar_fusion"],
        )

    if any(term in normalized for term in count_terms):
        return SupervisorDecision(
            classes=["SINGLE_IMAGE"],
            workflow=["visual_question_answering"],
            parameters={"intent": "counting"},
        )

    if any(term in normalized for term in grounding_terms) and any(
        term in normalized for term in object_terms
    ):
        return SupervisorDecision(
            classes=["GROUNDING_CAPTIONING"],
            workflow=["object_grounding_and_captioning"],
            parameters={"intent": "grounding"},
        )

    return SupervisorDecision(
        classes=["SINGLE_IMAGE"],
        workflow=["visual_question_answering"],
    )


def call_groq_supervisor(
    query: str,
    input_info: Dict[str, Any],
) -> SupervisorDecision:
    routed = route_query(query, int(input_info.get("image_count", 0)))
    if routed.classes != ["SINGLE_IMAGE"] or routed.parameters:
        return routed

    prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"INPUT CONFIGURATION:\n{json.dumps(input_info, indent=2)}\n\n"
        f"CLASSES:\n{json.dumps(CLASS_DESCRIPTIONS, indent=2)}\n"
        'Return exactly {"classes": [], "workflow": [], "parameters": {}}.'
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Groq at {GROQ_URL}.") from exc

    if response.status_code in {401, 402}:
        raise HTTPException(
            status_code=502,
            detail="Groq authentication or billing failed.",
        )
    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Groq returned HTTP {response.status_code}: {response.text[:500]}",
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        decision_json = json.loads(content)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Groq returned an invalid classification response.",
        ) from exc

    classes = [
        item for item in decision_json.get("classes", [])
        if item in VALID_CLASSES
    ]
    return SupervisorDecision(
        classes=classes or ["SINGLE_IMAGE"],
        workflow=decision_json.get("workflow", []),
        parameters=decision_json.get("parameters", {}),
    )
