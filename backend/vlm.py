
from typing import List, Dict, Any
from PIL import Image
import os
import re
import time
import base64
import io
import logging
import torch
from fastapi import HTTPException

MAX_NEW_TOKENS = int(
    os.getenv(
        "MAX_NEW_TOKENS",
        "256",
    )
)

from model_loader import (
    model,
    processor,
    MODEL_REGISTRY,
)

logger = logging.getLogger(__name__)

WORKFLOW_IMAGE_SIZE = {
    "GROUNDING_CAPTIONING": (512, 512),
    "SINGLE_IMAGE": (448, 448),
    "CHANGE_ANALYSIS": (512, 512),
    "OPTICAL_SAR": (224, 224),
    "GENERAL_REMOTE_SENSING": (448, 448),
}

# ============================================================
# WORKFLOW → ADAPTER
# ============================================================

WORKFLOW_TO_ADAPTER = {

    "SINGLE_IMAGE":
        "vqa",

    "GROUNDING_CAPTIONING":
        "captioning",

    "CHANGE_ANALYSIS":
        "change",

    "OPTICAL_SAR":
        "optical_sar",

    "GENERAL_REMOTE_SENSING":
        "vqa",
}


# ============================================================
# WORKFLOW → REGISTRY ENTRY
# ============================================================

WORKFLOW_TO_REGISTRY = {

    "SINGLE_IMAGE":
        "single_image_VQA_vlm",

    "GROUNDING_CAPTIONING":
        "grounding_captioning_vlm",

    "CHANGE_ANALYSIS":
        "change_analysis_vlm",

    "OPTICAL_SAR":
        "optical_sar_analysis_vlm",

    "GENERAL_REMOTE_SENSING":
        "single_image_VQA_vlm",
}


def extract_grounding_boxes(
    text: str,
    image_size: tuple[int, int] | None = None,
) -> List[List[float]]:
    """Extract boxes and normalize pixel coordinates when image dimensions exist."""
    pattern = r"(?:\[|\()\s*([0-9]*\.?[0-9]+)\s*[,\s]+\s*([0-9]*\.?[0-9]+)\s*[,;]\s*([0-9]*\.?[0-9]+)\s*[,\s]+\s*([0-9]*\.?[0-9]+)\s*(?:\]|\))"
    boxes = []
    for match in re.findall(pattern, text):
        box = [max(0.0, min(1.0, float(value))) for value in match]
        if image_size and max(float(value) for value in match) > 1:
            width, height = image_size
            x_min, y_min, x_max, y_max = [float(value) for value in match]
            box = [
                max(0.0, min(1.0, x_min / width)),
                max(0.0, min(1.0, y_min / height)),
                max(0.0, min(1.0, x_max / width)),
                max(0.0, min(1.0, y_max / height)),
            ]
        x_min, y_min, x_max, y_max = box
        if x_max > x_min and y_max > y_min:
            boxes.append(box)
    return boxes


def clean_generated_text(text: str) -> str:
    """Remove chat-template echoes so users see only the specialist answer."""
    cleaned = text.strip()
    if "assistant" in cleaned.lower():
        cleaned = re.split(r"assistant\s*", cleaned, flags=re.IGNORECASE)[-1]
    cleaned = re.sub(r"</?(answer|caption)>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def resize_for_workflow(images: List[Image.Image], workflow: str) -> List[Image.Image]:
    target_size = WORKFLOW_IMAGE_SIZE[workflow]
    resized = [image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS) for image in images]
    logger.info(
        "workflow_resize workflow=%s target=%sx%s input_sizes=%s output_count=%s",
        workflow,
        target_size[0],
        target_size[1],
        [image.size for image in images],
        len(resized),
    )
    return resized


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ============================================================
# VLM INFERENCE
# ============================================================

def run_vlm(
    images: List[Image.Image],
    query: str,
    workflow: str,
    grounding_requested: bool = False,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Select already-loaded adapter
    # --------------------------------------------------------

    if workflow not in WORKFLOW_TO_ADAPTER:

        raise ValueError(
            f"Unknown workflow: {workflow}"
        )

    adapter_name = WORKFLOW_TO_ADAPTER[
        workflow
    ]

    registry_name = WORKFLOW_TO_REGISTRY[
        workflow
    ]

    images = resize_for_workflow(images, workflow)
    logger.info(
        "adapter_selected workflow=%s adapter=%s registry=%s",
        workflow,
        adapter_name,
        registry_name,
    )

    # This does NOT download anything.
    # It simply activates one adapter already in memory.
    model.set_adapter(adapter_name)


    # --------------------------------------------------------
    # Workflow-specific prompt
    # --------------------------------------------------------

    if workflow == "GROUNDING_CAPTIONING":

        if any(term in query.lower() for term in ("describe", "caption", "summarize", "what do you see", "scene")):
            instruction = f"""
        You are a remote-sensing image-captioning specialist.

        Write a concise, polished description of the complete satellite scene.
        Mention dominant land cover, major structures, roads, vegetation,
        water, and spatial relationships only when visible.

        User request:
        {query}

        Return only the final scene caption and visual evidence.
        """
        else:
            instruction = f"""
        You are a remote-sensing object-grounding specialist.

        Locate the object or objects requested by the user in the image.
        Return exactly one line in this format first:
        BBOXES: [[x_min, y_min, x_max, y_max], ...]
        Coordinates must be normalized numbers between 0 and 1.
        Then write one short sentence describing the located object.
        Never omit the BBOXES line. If uncertain, return BBOXES: [].

        User request:
        {query}

        Return only the bounding boxes and a short evidence statement.
        """

    elif workflow == "SINGLE_IMAGE":

        instruction = f"""
        You are a remote-sensing vision-language specialist.

        Analyze the provided satellite image.

        User request:
        {query}

        Perform the requested task using visual evidence.

        Return:

        Answer:
        <answer>

        Evidence:
        <important visual evidence>

        Confidence:
        <high / medium / low>

        Do not claim information that cannot be observed.
        """

    elif workflow == "CHANGE_ANALYSIS":

        instruction = f"""
        You are a remote-sensing change-analysis specialist.

        Two images of the same geographic area are provided.

        Image 1 = earlier observation.
        Image 2 = later observation.

        User request:
        {query}

        Compare the two observations.

        Identify:
        1. What changed
        2. Where the change occurred
        3. Approximate nature of the change
        4. Evidence supporting your conclusion
        5. Confidence

        Return:

        Answer:
        <answer>

        Changes:
        <list of changes>

        Evidence:
        <visual evidence>

        Confidence:
        <high / medium / low>
        """

    elif workflow == "OPTICAL_SAR":

        instruction = f"""
        You are a remote-sensing optical-SAR fusion specialist.

        Two complementary observations are provided.

        Image 1:
        Optical/multispectral

        Image 2:
        SAR

        User request:
        {query}

        Use BOTH observations.

        Explain complementary evidence from:
        - optical spectral/contextual appearance
        - SAR structural/backscatter information

        Return:

        Answer:
        <answer>

        Optical Evidence:
        <evidence>

        SAR Evidence:
        <evidence>

        Combined Interpretation:
        <interpretation>

        Confidence:
        <high / medium / low>
        """

    else:

        instruction = f"""
        You are a remote-sensing specialist.

        Analyze the provided image(s) and answer:

        {query}

        Ground the the answer in observable evidence.

        Return:

        Answer:
        <answer>

        Evidence:
        <evidence>

        Confidence:
        <high / medium / low>
        """


    # --------------------------------------------------------
    # Build Qwen conversation
    # --------------------------------------------------------

    content = []

    for image in images:

        content.append(
            {
                "type": "image",
                "image": image,
            }
        )

    content.append(
        {
            "type": "text",
            "text": instruction,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]


    # --------------------------------------------------------
    # Apply chat template
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    # --------------------------------------------------------
    # Prepare inputs
    # --------------------------------------------------------

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(model.device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    start = time.time()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    elapsed = time.time() - start


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    raw_generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]
    generated_text = clean_generated_text(raw_generated_text)
    logger.info(
        "specialist_output workflow=%s text=%r",
        workflow,
        generated_text[:1000],
    )

    grounding_boxes = extract_grounding_boxes(generated_text, images[0].size) if grounding_requested else []


    return {

        "answer": generated_text,

        "adapter": adapter_name,

        "specialist": registry_name,

        "caption": generated_text if workflow == "GROUNDING_CAPTIONING" else None,

        "is_grounding": grounding_requested,

        "boxes": grounding_boxes,

        "evidence_image": image_data_url(images[0]) if grounding_requested else None,

        "latency_seconds": round(
            elapsed,
            3,
        ),
    }


# ============================================================
# EXECUTE WORKFLOW
# ============================================================

def execute_workflow(
    classes: List[str],
    images: List[Image.Image],
    query: str,
    grounding_requested: bool = False,
) -> Dict[str, Any]:

    outputs = []
    logger.info("workflow_execution_started classes=%s image_count=%s", classes, len(images))

    # --------------------------------------------------------
    # SINGLE IMAGE
    # --------------------------------------------------------

    if "SINGLE_IMAGE" in classes:

        result = run_vlm(
            images=images[:1],
            query=query,
            workflow="SINGLE_IMAGE",
        )

        outputs.append(
            {
                "task": "SINGLE_IMAGE",
                "model": MODEL_REGISTRY[
                    "single_image_VQA_vlm"
                ],
                "result": result,
            }
        )

    if "GROUNDING_CAPTIONING" in classes:

        result = run_vlm(
            images=images[:1],
            query=query,
            workflow="GROUNDING_CAPTIONING",
            grounding_requested=grounding_requested,
        )

        outputs.append(
            {
                "task": "GROUNDING_CAPTIONING",
                "model": MODEL_REGISTRY[
                    "grounding_captioning_vlm"
                ],
                "result": result,
            }
        )

    # --------------------------------------------------------
    # CHANGE ANALYSIS
    # --------------------------------------------------------

    if "CHANGE_ANALYSIS" in classes:

        if len(images) < 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "CHANGE_ANALYSIS requires "
                    "two images."
                ),
            )

        result = run_vlm(
            images=images[:2],
            query=query,
            workflow="CHANGE_ANALYSIS",
        )

        outputs.append(
            {
                "task": "CHANGE_ANALYSIS",
                "model": MODEL_REGISTRY[
                    "change_analysis_vlm"
                ],
                "result": result,
            }
        )


    # --------------------------------------------------------
    # OPTICAL + SAR
    # --------------------------------------------------------

    if "OPTICAL_SAR" in classes:

        if len(images) < 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "OPTICAL_SAR requires "
                    "two images."
                ),
            )

        result = run_vlm(
            images=images[:2],
            query=query,
            workflow="OPTICAL_SAR",
        )

        outputs.append(
            {
                "task": "OPTICAL_SAR",
                "model": MODEL_REGISTRY[
                    "optical_sar_analysis_vlm"
                ],
                "result": result,
            }
        )


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    if (
        "GENERAL_REMOTE_SENSING" in classes
        and not outputs
    ):

        result = run_vlm(
            images=images,
            query=query,
            workflow="GENERAL_REMOTE_SENSING",
        )

        outputs.append(
            {
                "task": "GENERAL_REMOTE_SENSING",
                "model": MODEL_REGISTRY[
                    "grounding_captioning_vlm"
                ],
                "result": result,
            }
        )


    logger.info("workflow_execution_completed tasks=%s", [output["task"] for output in outputs])
    return {
        "outputs": outputs
    }