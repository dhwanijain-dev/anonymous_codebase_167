import os
import time
import uuid
import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("satquery.api")

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from supervisor import call_groq_supervisor
from rl_router import route_with_rl

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

# Whether to use the trained RL router (True) or the keyword heuristic (False).
# Set SATQUERY_USE_RL_ROUTER=false to force keyword routing even when a
# checkpoint is present.
USE_RL_ROUTER = os.getenv("SATQUERY_USE_RL_ROUTER", "true").lower() == "true"


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
    router_checkpoint = os.getenv(
        "ROUTER_CHECKPOINT",
        str(Path(__file__).parent / "satquery_router.pt"),
    )
    return {
        "status": "ok",

        "base_model": BASE_MODEL,

        "supervisor": GROQ_MODEL,

        "router_type": "rl" if (USE_RL_ROUTER and Path(router_checkpoint).exists()) else "keyword",

        "router_checkpoint": router_checkpoint if Path(router_checkpoint).exists() else None,

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
    """
    Lightweight classification endpoint — no image upload needed.
    Uses keyword routing (RL router requires real image bytes).
    """
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
        "router": decision.router,
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

    logger.info("analysis_request_received query=%r modalities=%s", query, modalities)

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

    logger.info("analysis_inputs_validated images=%s metadata=%s", len(decoded_images), image_metadata)


    # ========================================================
    # ROUTER  (RL or keyword fallback)
    # ========================================================

    supervisor_start = time.time()

    if USE_RL_ROUTER:
        # Write decoded PIL images to a temp directory so the RL feature
        # encoder can open them from disk (it calls Image.open internally).
        tmp_dir = tempfile.mkdtemp(prefix="satquery_router_")
        tmp_image_paths: List[str] = []
        modalities_list: List[str] = []
        try:
            for idx, (pil_img, meta) in enumerate(zip(decoded_images, image_metadata)):
                suffix = meta["format"] if meta["format"] in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} else ".png"
                tmp_path = str(Path(tmp_dir) / f"img{idx}{suffix}")
                pil_img.save(tmp_path)
                tmp_image_paths.append(tmp_path)
                modalities_list.append(modalities if len(decoded_images) == 1 else "optical")

            decision = route_with_rl(
                query=query,
                image_paths=tmp_image_paths,
                modalities=modalities_list,
            )
        except Exception as exc:
            logger.error("rl_router failed (%s), falling back to keyword", exc, exc_info=True)
            decision = call_groq_supervisor(
                query=query,
                input_info={"image_count": len(decoded_images), "modalities": modalities},
            )
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        decision = call_groq_supervisor(
            query=query,
            input_info={"image_count": len(decoded_images), "modalities": modalities},
        )

    logger.info(
        "router_decision router=%s classes=%s workflow=%s parameters=%s",
        decision.router,
        decision.classes,
        decision.workflow,
        decision.parameters,
    )

    supervisor_latency = time.time() - supervisor_start


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

        grounding_requested=decision.parameters.get("intent") == "grounding",
    )

    logger.info("analysis_completed request_id=%s tasks=%s", request_id, [task["task"] for task in execution["outputs"]])


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
                    "router",

                "type":
                    decision.router,

                "provider":
                    "SatQuery RL" if decision.router == "rl" else "keyword-heuristic",

                "latency_seconds":
                    round(
                        supervisor_latency,
                        3,
                    ),

                "reasoning":
                    decision.reasoning,
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
        answers = [
            task["result"]["answer"]
            for task in execution["outputs"]
            if task["result"].get("answer")
        ]
        if len(answers) > 1 and decision.parameters.get("combine_outputs"):
            first_answer = (
                "Visual analysis:\n"
                f"{answers[0].strip()}\n\n"
                "Scene caption:\n"
                f"{answers[1].strip()}"
            )
        elif answers:
            first_answer = answers[0]


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

    if USE_RL_ROUTER:
        tmp_dir = tempfile.mkdtemp(prefix="satquery_router_")
        tmp_paths: List[str] = []
        try:
            for idx, (pil_img, img_path) in enumerate(zip(decoded_images, image_paths)):
                suffix = Path(img_path).suffix or ".png"
                tmp_p = str(Path(tmp_dir) / f"img{idx}{suffix}")
                pil_img.save(tmp_p)
                tmp_paths.append(tmp_p)
            decision = route_with_rl(
                query=query,
                image_paths=tmp_paths,
                modalities=[modalities] * len(decoded_images),
            )
        except Exception as exc:
            logger.error("rl_router cli failed (%s), falling back", exc, exc_info=True)
            decision = call_groq_supervisor(
                query=query,
                input_info={"image_count": len(decoded_images), "modalities": modalities},
            )
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        decision = call_groq_supervisor(
            query=query,
            input_info={"image_count": len(decoded_images), "modalities": modalities},
        )

    execution = execute_workflow(
        classes=decision.classes,
        images=decoded_images,
        query=query,
        grounding_requested=decision.parameters.get("intent") == "grounding",
    )

    return {
        "answer": execution["outputs"][0]["result"]["answer"] if execution["outputs"] else None,
        "tasks": execution["outputs"],
        "classes": decision.classes,
        "workflow": decision.workflow,
        "router": decision.router,
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