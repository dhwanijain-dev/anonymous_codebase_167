import os
import time
import uuid
import argparse
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from supervisor import call_groq_supervisor

from image_validation import (
    validate_extension,
    load_image_bytes,
)

from vlm import execute_workflow

from model_loader import (
    model,
    processor,
    MODEL_REGISTRY,
    BASE_MODEL,
)


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

MAX_NEW_TOKENS = int(
    os.getenv(
        "MAX_NEW_TOKENS",
        "256",
    )
)

USE_4BIT = (
    os.getenv(
        "USE_4BIT",
        "true",
    ).lower()
    == "true"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SatQuery AI",
    description="Agentic remote-sensing VLM backend",
    version="1.0.0",
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "base_model": BASE_MODEL,

        "supervisor": GROQ_MODEL,

        "device": str(
            model.device
        ),
    }


# ============================================================
# AVAILABLE MODELS
# ============================================================

@app.get("/models")
def models():

    return {
        "base_model": BASE_MODEL,

        "models": MODEL_REGISTRY,
    }


# ============================================================
# SUPERVISOR-ONLY ENDPOINT
# ============================================================

@app.post("/classify")
async def classify_query(
    query: str = Form(...),

    image_count: int = Form(...),

    modalities: str = Form(
        "unknown"
    ),
):

    input_info = {

        "image_count": image_count,

        "modalities": modalities,
    }

    decision = call_groq_supervisor(
        query=query,

        input_info=input_info,
    )

    return {

        "query": query,

        "classes": decision.classes,

        "workflow": decision.workflow,

        "parameters": decision.parameters,
    }


# ============================================================
# MAIN ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(
    query: str = Form(...),

    image1: UploadFile = File(..., description="Primary image (.tif, .tiff, .png, .jpg, or .jpeg)"),

    image2: UploadFile | None = File(
        None,
        description="Optional second image for change or optical-SAR analysis",
    ),

    modalities: str = Form(
        "unknown"
    ),
):

    images = [image1]
    if image2 is not None:
        images.append(image2)

    request_id = str(
        uuid.uuid4()
    )

    start_time = time.time()


    # ========================================================
    # IMAGE COUNT VALIDATION
    # ========================================================

    if len(images) < 1:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one image "
                "is required."
            ),
        )


    if len(images) > 2:

        raise HTTPException(
            status_code=400,
            detail=(
                "SatQuery currently "
                "supports one or two images."
            ),
        )


    # ========================================================
    # READ + VALIDATE IMAGES
    # ========================================================

    decoded_images = []

    image_metadata = []


    for upload in images:

        if not upload.filename:

            raise HTTPException(
                status_code=400,
                detail="Filename is missing.",
            )


        validate_extension(
            upload.filename
        )


        data = await upload.read()


        if not data:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Empty file: "
                    f"{upload.filename}"
                ),
            )


        image = load_image_bytes(
            data,

            upload.filename,
        )


        decoded_images.append(
            image
        )


        image_metadata.append(
            {
                "filename":
                    upload.filename,

                "format":
                    Path(
                        upload.filename
                    ).suffix.lower(),

                "width":
                    image.width,

                "height":
                    image.height,
            }
        )


    # ========================================================
    # SUPERVISOR
    # ========================================================

    supervisor_input = {

        "image_count":
            len(decoded_images),

        "modalities":
            modalities,

        "images":
            image_metadata,
    }


    supervisor_start = time.time()


    decision = call_groq_supervisor(
        query=query,

        input_info=supervisor_input,
    )


    supervisor_latency = (
        time.time()
        - supervisor_start
    )


    # ========================================================
    # INPUT / WORKFLOW COMPATIBILITY
    # ========================================================

    if (
        "CHANGE_ANALYSIS"
        in decision.classes

        and len(decoded_images) != 2
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "CHANGE_ANALYSIS requires "
                "exactly two images."
            ),
        )


    if (
        "OPTICAL_SAR"
        in decision.classes

        and len(decoded_images) != 2
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "OPTICAL_SAR requires "
                "exactly two images."
            ),
        )


    # ========================================================
    # EXECUTE SPECIALIST WORKFLOW
    # ========================================================

    execution = execute_workflow(
        classes=decision.classes,

        images=decoded_images,

        query=query,
    )


    total_latency = (
        time.time()
        - start_time
    )


    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    execution_trace = {

        "request_id":
            request_id,

        "query":
            query,

        "selected_tasks":
            decision.classes,

        "workflow":
            decision.workflow,

        "parameters":
            decision.parameters,

        "models": [

            {
                "role":
                    "supervisor",

                "provider":
                    "Groq",

                "model":
                    GROQ_MODEL,

                "latency_seconds":
                    round(
                        supervisor_latency,
                        3,
                    ),
            },

            {
                "role":
                    "specialist",

                "provider":
                    "Hugging Face",

                "base_model":
                    BASE_MODEL,

                "adapters_loaded":
                    list(
                        MODEL_REGISTRY.keys()
                    ),
            },
        ],

        "inputs":
            image_metadata,

        "total_latency_seconds":
            round(
                total_latency,
                3,
            ),
    }


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    first_answer = None

    if execution["outputs"]:

        first_answer = (
            execution["outputs"][0]
            ["result"]
            ["answer"]
        )


    return {

        "request_id":
            request_id,

        "answer":
            first_answer,

        "tasks":
            execution["outputs"],

        "execution_trace":
            execution_trace,
    }


def run_cli_analysis(image_paths: List[str], query: str, modalities: str):
    if len(image_paths) > 2:
        raise ValueError("SatQuery supports one or two images.")

    decoded_images = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        validate_extension(path.name)
        decoded_images.append(load_image_bytes(path.read_bytes(), path.name))

    decision = call_groq_supervisor(
        query=query,
        input_info={
            "image_count": len(decoded_images),
            "modalities": modalities,
        },
    )

    execution = execute_workflow(
        classes=decision.classes,
        images=decoded_images,
        query=query,
    )

    return {
        "answer": execution["outputs"][0]["result"]["answer"] if execution["outputs"] else None,
        "tasks": execution["outputs"],
        "classes": decision.classes,
        "workflow": decision.workflow,
    }


def parse_cli_args():
    parser = argparse.ArgumentParser(description="SatQuery API or local image analysis")
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        help="Local image path; pass twice for paired analysis",
    )
    parser.add_argument("--query", help="Question to ask about the image")
    parser.add_argument("--modalities", default="unknown", help="Input modality label")
    return parser.parse_args()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn
    cli_args = parse_cli_args()

    if cli_args.images:
        if not cli_args.query:
            raise SystemExit("--query is required when using --image")

        try:
            print(json.dumps(
                run_cli_analysis(cli_args.images, cli_args.query, cli_args.modalities),
                indent=2,
            ))
        except (FileNotFoundError, ValueError, HTTPException) as exc:
            raise SystemExit(str(exc)) from exc
        raise SystemExit(0)

    uvicorn.run(
        app,

        host="0.0.0.0",

        port=8000,

        reload=False,
    )