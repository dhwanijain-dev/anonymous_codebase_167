import os

from unsloth import FastVisionModel
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = os.getenv(
    "BASE_MODEL",
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
)

HF_MODEL_REPO1 = os.getenv(
    "HF_MODEL_REPO1",
    "whrecker/satquery-single-image-vqa",
)

HF_MODEL_REPO2 = os.getenv(
    "HF_MODEL_REPO2",
    "whrecker/satquery-grounding-captioning",
)

HF_MODEL_REPO3 = os.getenv(
    "HF_MODEL_REPO3",
    "whrecker/satquery-change-analysis",
)

HF_MODEL_REPO4 = os.getenv(
    "HF_MODEL_REPO4",
    "whrecker/satquery-optical-sar",
)


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
)

print("✅ Captioning/grounding adapter loaded")


print("Loading change-analysis adapter...")

model.load_adapter(
    HF_MODEL_REPO3,
    adapter_name="change",
    is_trainable=False,
)

print("✅ Change-analysis adapter loaded")


print("Loading optical-SAR adapter...")

model.load_adapter(
    HF_MODEL_REPO4,
    adapter_name="optical_sar",
    is_trainable=False,
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