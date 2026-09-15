
from typing import List, Dict, Any
from PIL import Image

import time
import torch
from model_loader import (
    model,
    processor,
    MODEL_REGISTRY,
)

# ============================================================
# WORKFLOW → ADAPTER
# ============================================================

WORKFLOW_TO_ADAPTER = {

    "SINGLE_IMAGE":
        "vqa",

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

    "CHANGE_ANALYSIS":
        "change_analysis_vlm",

    "OPTICAL_SAR":
        "optical_sar_analysis_vlm",

    "GENERAL_REMOTE_SENSING":
        "single_image_VQA_vlm",
}


# ============================================================
# VLM INFERENCE
# ============================================================

def run_vlm(
    images: List[Image.Image],
    query: str,
    workflow: str,
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

    # This does NOT download anything.
    # It simply activates one adapter already in memory.
    model.set_adapter(adapter_name)


    # --------------------------------------------------------
    # Workflow-specific prompt
    # --------------------------------------------------------

    if workflow == "SINGLE_IMAGE":

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

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]


    return {

        "answer": generated_text,

        "adapter": adapter_name,

        "specialist": registry_name,

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
) -> Dict[str, Any]:

    outputs = []

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


    return {
        "outputs": outputs
    }