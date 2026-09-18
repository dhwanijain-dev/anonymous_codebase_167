import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face reads these settings during import, so configure them first.
# Set HF_TOKEN_BASE in the shell; HF_TOKEN and HF_TOKEN1 remain supported fallbacks.
hf_token = os.getenv("HF_TOKEN_BASE") or os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN1")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
# HTTP is more reliable than Xet for this multi-gigabyte model on unstable links.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from unsloth import FastVisionModel
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

# Set one token per Hugging Face repository before starting the backend:
# export HF_TOKEN_BASE="hf_base_model_token"
# export HF_TOKEN1="hf_single_image_token"
# export HF_TOKEN2="hf_grounding_token"
# export HF_TOKEN3="hf_change_analysis_token"
# export HF_TOKEN4="hf_optical_sar_token"
# For high-performance Xet downloads, use HF_XET_HIGH_PERFORMANCE=1 instead
# of the deprecated HF_HUB_ENABLE_HF_TRANSFER variable.

BASE_MODEL = os.getenv(
    "BASE_MODEL",
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
)

HF_MODEL_REPO1 = os.getenv(
    "HF_MODEL_REPO1",
    "whrecker/qwen2.5-3b-vl-bigearthnet-txt-lora",
)

HF_MODEL_REPO2 = os.getenv(
    "HF_MODEL_REPO2",
    "tutuk45/satquery-grounding-qwen2.5vl-3b-lora",
)

HF_MODEL_REPO3 = os.getenv(
    "HF_MODEL_REPO3",
    "yashord/qwen2.5-3b-vl-changevqa-lora-v2",
)

HF_MODEL_REPO4 = os.getenv(
    "HF_MODEL_REPO4",
    "whrecker/qwen2.5-3b-vl-bigearthnet-txt-lora",
)

HF_TOKEN_BASE = os.getenv("HF_TOKEN_BASE") or os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN1")
HF_TOKEN1 = os.getenv("HF_TOKEN1") or HF_TOKEN_BASE
HF_TOKEN2 = os.getenv("HF_TOKEN2") or HF_TOKEN_BASE
HF_TOKEN3 = os.getenv("HF_TOKEN3") or HF_TOKEN_BASE
HF_TOKEN4 = os.getenv("HF_TOKEN4") or HF_TOKEN_BASE


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {

    "single_image_VQA_vlm": {
        "adapter_name": "vqa",
        "adapter_repo": HF_MODEL_REPO1,
        "base_model": BASE_MODEL,
        "type": "Qwen2.5-VL + LoRA",
        "source": "Hugging Face",
        "tasks": [
            "single_image_vqa",
        ],
    },

    "grounding_captioning_vlm": {
        "adapter_name": "captioning",
        "adapter_repo": HF_MODEL_REPO2,
        "base_model": BASE_MODEL,
        "type": "Qwen2.5-VL + LoRA",
        "source": "Hugging Face",
        "tasks": [
            "captioning",
            "grounding",
        ],
    },

    "change_analysis_vlm": {
        "adapter_name": "change",
        "adapter_repo": HF_MODEL_REPO3,
        "base_model": BASE_MODEL,
        "type": "Qwen2.5-VL + LoRA",
        "source": "Hugging Face",
        "tasks": [
            "change_analysis",
        ],
    },

    "optical_sar_analysis_vlm": {
        "adapter_name": "optical_sar",
        "adapter_repo": HF_MODEL_REPO4,
        "base_model": BASE_MODEL,
        "type": "Qwen2.5-VL + LoRA",
        "source": "Hugging Face",
        "tasks": [
            "optical_sar_analysis",
        ],
    },
}


# ============================================================
# ADAPTER MAP
# ============================================================

ADAPTERS = {
    "single_image_VQA_vlm": HF_MODEL_REPO1,
    "grounding_captioning_vlm": HF_MODEL_REPO2,
    "change_analysis_vlm": HF_MODEL_REPO3,
    "optical_sar_analysis_vlm": HF_MODEL_REPO4,
}


# ============================================================
# LOAD BASE MODEL ONCE
# ============================================================

print("=" * 60)
print("Loading Qwen2.5-VL base model")
print("=" * 60)

model, processor = FastVisionModel.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    token=HF_TOKEN_BASE,
)

print("✅ Base model loaded")


# ============================================================
# LOAD FIRST ADAPTER
# ============================================================

print("Loading VQA adapter...")

model = PeftModel.from_pretrained(
    model,
    HF_MODEL_REPO1,
    adapter_name="vqa",
    is_trainable=False,
    token=HF_TOKEN1,
)

print("✅ VQA adapter loaded")


# ============================================================
# LOAD REMAINING ADAPTERS
# ============================================================

print("Loading captioning/grounding adapter...")

model.load_adapter(
    HF_MODEL_REPO2,
    adapter_name="captioning",
    is_trainable=False,
    token=HF_TOKEN2,
)

print("✅ Captioning/grounding adapter loaded")


print("Loading change-analysis adapter...")

model.load_adapter(
    HF_MODEL_REPO3,
    adapter_name="change",
    is_trainable=False,
    token=HF_TOKEN3,
)

print("✅ Change-analysis adapter loaded")


print("Loading optical-SAR adapter...")

model.load_adapter(
    HF_MODEL_REPO4,
    adapter_name="optical_sar",
    is_trainable=False,
    token=HF_TOKEN4,
)

print("✅ Optical-SAR adapter loaded")


# ============================================================
# INFERENCE MODE
# ============================================================

FastVisionModel.for_inference(model)

# Default adapter
model.set_adapter("vqa")


print("=" * 60)
print("✅ ALL FOUR ADAPTERS LOADED")
print("Available adapters:", list(ADAPTERS.keys()))
print("Active adapter: vqa")
print("=" * 60)