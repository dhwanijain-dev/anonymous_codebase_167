from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
class SupervisorDecision(BaseModel):
    classes: List[str] = Field(default_factory=list)

    reasoning: Optional[str] = None

    workflow: List[str] = Field(default_factory=list)

    parameters: Dict[str, Any] = Field(
        default_factory=dict
    )

CLASS_DESCRIPTIONS = {

    "SINGLE_IMAGE": """
Single-image remote sensing analysis.

Use this for:
- visual question answering
- image description
- scene understanding
- land-cover identification
- object identification
- captioning
- text-guided grounding

Input:
Exactly one image.
""",

    "CHANGE_ANALYSIS": """
Bi-temporal remote sensing change analysis.

Use this when:
- two images show the same geographic region
- images correspond to different dates
- user asks what changed
- user asks whether something increased/decreased
- change description is required
- change-based VQA is requested

Input:
Exactly two spatially corresponding images acquired at different times.
""",

    "OPTICAL_SAR": """
Cross-modal optical-SAR analysis.

Use this when:
- one optical/multispectral image is provided
- one SAR image is provided
- both describe the same geographic area
- the user asks for joint/complementary analysis
- the query involves built-up, water, vegetation, infrastructure, etc.
using both modalities

Input:
Optical/multispectral + SAR image pair.
""",

    "GENERAL_REMOTE_SENSING": """
General remote sensing analysis that does not clearly belong
to a specific paired workflow.
"""
}


VALID_CLASSES = set(CLASS_DESCRIPTIONS.keys())


SUPERVISOR_SYSTEM_PROMPT = """
You are the supervisor agent of SatQuery AI.

Your job is NOT to answer the remote-sensing question.

Your job is to classify the user's request and determine
which specialist workflow(s) should be executed.

You may select MULTIPLE classes.

The available classes are:

SINGLE_IMAGE
CHANGE_ANALYSIS
OPTICAL_SAR
GENERAL_REMOTE_SENSING

Definitions:

SINGLE_IMAGE:
One image is used for VQA, captioning, scene description,
object identification, or text-guided grounding.

CHANGE_ANALYSIS:
Two corresponding images represent different times/dates
and the user asks about change.

OPTICAL_SAR:
An optical/multispectral image and a SAR image describe the
same location and the user asks for joint/cross-modal analysis.

GENERAL_REMOTE_SENSING:
Remote-sensing analysis that does not fit the above workflows.

Important:

- Do NOT answer the user's question.
- Do NOT invent models.
- Return JSON only.
- classes may contain more than one class.
- workflow should contain the ordered specialist actions.
- If two classes are required, include both.
"""

def call_ollama_supervisor(
    query: str,
    input_info: Dict[str, Any],
) -> SupervisorDecision:

    prompt = f"""
USER QUERY:
{query}

INPUT CONFIGURATION:
{json.dumps(input_info, indent=2)}

AVAILABLE CLASS DEFINITIONS:
{json.dumps(CLASS_DESCRIPTIONS, indent=2)}

Return exactly this JSON structure:

{{
    "classes": [],
    "workflow": [],
    "parameters": {{}}
}}

Do not include markdown.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SUPERVISOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    content = data["message"]["content"]

    try:
        decision_json = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Supervisor returned invalid JSON: {content}"
        ) from exc

    classes = [
        c for c in decision_json.get("classes", [])
        if c in VALID_CLASSES
    ]

    if not classes:
        classes = ["GENERAL_REMOTE_SENSING"]

    workflow = decision_json.get(
        "workflow",
        []
    )

    return SupervisorDecision(
        classes=classes,
        workflow=workflow,
        parameters=decision_json.get(
            "parameters",
            {}
        ),
    )