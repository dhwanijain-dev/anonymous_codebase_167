import os
import time
import uuid
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from supervisor import call_ollama_supervisor

from image_validataion import (
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

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b",
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

        "supervisor": OLLAMA_MODEL,

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

    decision = call_ollama_supervisor(
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

    images: List[UploadFile] = File(...),

    modalities: str = Form(
        "unknown"
    ),
):

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


    decision = call_ollama_supervisor(
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
                    "Ollama",

                "model":
                    OLLAMA_MODEL,

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


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,

        host="0.0.0.0",

        port=8000,

        reload=False,
    )