import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoImageProcessor
import torch 
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

import torch
import torch.nn as nn
from unsloth import FastVisionModel
from peft import PeftModel
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoImageProcessor


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
    "whrecker/qwen2.5-3b-vl-rsvqa-hr-lora",
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
    "whrecker/qwen2.5-3b-vl-bigearthnet-sar-fusion-experiment",
)

HF_TOKEN_BASE = os.getenv("HF_TOKEN_BASE") or os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN1")
HF_TOKEN1 = os.getenv("HF_TOKEN1") or HF_TOKEN_BASE
HF_TOKEN2 = os.getenv("HF_TOKEN2") or HF_TOKEN_BASE
HF_TOKEN3 = os.getenv("HF_TOKEN3") or HF_TOKEN_BASE
HF_TOKEN4 = os.getenv("HF_TOKEN4") or HF_TOKEN_BASE

# Must match the fusion-experiment notebook's training-time config exactly —
# the projector's weight shapes and the SAR patch-token count depend on these.
SAR_ENCODER_ID = "BiliSakura/SARMAE-transformers"
SAR_ENCODER_SUBFOLDER = "vit-base-patch16-pretrain"
SAR_IMAGE_SIZE = 224
SAR_PATCH_SIZE = 16
SAR_NUM_PATCHES = (SAR_IMAGE_SIZE // SAR_PATCH_SIZE) ** 2  # 196
NUM_SAR_TOKENS = 16


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
        # Not a plain LoRA adapter — needs the SARMAE encoder + projector
        # attached separately below, not just model.load_adapter(...).
        "type": "Qwen2.5-VL + LoRA + SARMAE encoder + SAR projector",
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
# ATTACH THE SARMAE ENCODER + PROJECTOR FOR THE OPTICAL-SAR ADAPTER
# ============================================================
# The "optical_sar" adapter alone is not the full model — training routed
# the SAR image through a separate SARMAE encoder + a trainable projector,
# then prepended those tokens ahead of the normal text+S2 sequence (see the
# fusion-experiment notebook). None of that is a PEFT adapter, so
# model.load_adapter() above never touched it. It has to be loaded and
# wired up separately here, exactly once.

print("Loading SARMAE encoder for the optical-SAR adapter...")


# Download the complete checkpoint, including custom modeling code.
sar_repo_path = snapshot_download(
    repo_id=SAR_ENCODER_ID,
    allow_patterns=[
        f"{SAR_ENCODER_SUBFOLDER}/*",
    ],
    token=HF_TOKEN4,
)

sar_model_path = os.path.join(
    sar_repo_path,
    SAR_ENCODER_SUBFOLDER,
)

print("SARMAE local path:", sar_model_path)

# Load from the local directory instead of resolving the remote subfolder.
sar_encoder = AutoModel.from_pretrained(
    sar_model_path,
    trust_remote_code=True,
    token=HF_TOKEN4,
)

device = model.device if hasattr(model, "device") else "cuda"

sar_encoder = sar_encoder.to(device)
sar_encoder.eval()

for p in sar_encoder.parameters():
    p.requires_grad_(False)

sar_image_processor = AutoImageProcessor.from_pretrained(
    sar_model_path,
    trust_remote_code=True,
    token=HF_TOKEN4,
)

sar_hidden_size = sar_encoder.config.hidden_size

print("✅ SARMAE encoder loaded")
print("Hidden size:", sar_hidden_size)

# --- Same attribute-discovery logic as the training notebook: reaches into
#     model internals that aren't guaranteed-stable public API, tries
#     several plausible paths, and fails loudly if none match. ---

def _find_first_attr(obj, candidate_paths):
    for path in candidate_paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            return cur, path
    return None, None


QWEN_VISUAL_MODULE, visual_path = _find_first_attr(model, [
    "visual",
    "model.visual",
    "base_model.model.visual",
    "base_model.model.model.visual",
])
print("Vision tower found at attribute path:", visual_path)
assert QWEN_VISUAL_MODULE is not None, (
    "Could not locate Qwen's vision tower automatically. Run print(model) "
    "and find the module that processes pixel_values (commonly 'visual'), "
    "then hardcode QWEN_VISUAL_MODULE to that path before serving requests."
)


def _find_image_token_id(m):
    cfg = getattr(m, "config", None)
    if cfg is not None and getattr(cfg, "image_token_id", None) is not None:
        return cfg.image_token_id
    for sub in ["text_config", "vision_config"]:
        sub_cfg = getattr(cfg, sub, None) if cfg is not None else None
        if sub_cfg is not None and getattr(sub_cfg, "image_token_id", None) is not None:
            return sub_cfg.image_token_id
    return None


IMAGE_TOKEN_ID = _find_image_token_id(model)
print("image_token_id:", IMAGE_TOKEN_ID)
assert IMAGE_TOKEN_ID is not None, (
    "Could not find image_token_id on the model config. Find it manually via "
    "processor.tokenizer.convert_tokens_to_ids('<|image_pad|>') and hardcode "
    "IMAGE_TOKEN_ID before serving requests."
)

TEXT_EMBED_LAYER = model.get_input_embeddings()

qwen_hidden_size = None
for path in ["config.hidden_size", "config.text_config.hidden_size"]:
    val, _ = _find_first_attr(model, [path])
    if isinstance(val, int):
        qwen_hidden_size = val
        break
assert qwen_hidden_size is not None, (
    "Could not find Qwen's hidden size automatically. Inspect model.config "
    "and hardcode qwen_hidden_size before serving requests."
)
print("Qwen hidden size:", qwen_hidden_size)


# --- Same projector architecture as training — must match exactly to load
#     the trained state_dict correctly. ---

class SARProjector(nn.Module):
    def __init__(self, sar_hidden, qwen_hidden, num_tokens):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.proj = nn.Sequential(
            nn.LayerNorm(sar_hidden),
            nn.Linear(sar_hidden, qwen_hidden),
            nn.GELU(),
            nn.Linear(qwen_hidden, qwen_hidden),
        )

    def forward(self, sar_patch_embeds):
        x = sar_patch_embeds.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return self.proj(x)


sar_projector = SARProjector(sar_hidden_size, qwen_hidden_size, NUM_SAR_TOKENS)
sar_projector = sar_projector.to(model.device if hasattr(model, "device") else "cuda")

projector_path = hf_hub_download(
    repo_id=HF_MODEL_REPO4,
    filename="sar_projector.pt",
    token=HF_TOKEN4,
)
sar_projector.load_state_dict(
    torch.load(projector_path, map_location=model.device if hasattr(model, "device") else "cuda")
)
sar_projector.eval()
for p in sar_projector.parameters():
    p.requires_grad_(False)

print("✅ SAR projector loaded from", HF_MODEL_REPO4)


# ============================================================
# FUSION FORWARD PASS (inference-time — no labels, feeds model.generate)
# ============================================================

def build_fused_inputs_embeds(input_ids, pixel_values, image_grid_thw, sar_embeds):
    """Same construction as training: Qwen's own text+S2 embeddings computed
    normally, then SAR tokens prepended ahead of them."""
    text_embeds = TEXT_EMBED_LAYER(input_ids)

    image_embeds = QWEN_VISUAL_MODULE(pixel_values, grid_thw=image_grid_thw)
    if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
        image_embeds = image_embeds.squeeze(0)

    image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
    num_image_tokens = int(image_mask.sum().item())
    assert num_image_tokens == image_embeds.shape[0], (
        f"Placeholder token count ({num_image_tokens}) != image embedding "
        f"count ({image_embeds.shape[0]}) — check IMAGE_TOKEN_ID / visual_path."
    )

    merged = text_embeds.clone()
    merged[0, image_mask] = image_embeds.to(merged.dtype)

    fused = torch.cat([sar_embeds.to(merged.dtype), merged], dim=1)
    return fused


@torch.no_grad()
def run_optical_sar_inference(image_s2, image_s1, question, max_new_tokens=256):
    """
    image_s2: PIL Image, optical (fed through Qwen's own vision tower)
    image_s1: PIL Image, SAR (fed through SARMAE)
    question: str
    """
    model.set_adapter("optical_sar")
    FastVisionModel.for_inference(model)

    device = model.device if hasattr(model, "device") else "cuda"

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image_s2], return_tensors="pt").to(device)

    sar_pixel = sar_image_processor(images=image_s1, return_tensors="pt")
    sar_pixel = {k: v.to(device) for k, v in sar_pixel.items()}
    sar_out = sar_encoder(**sar_pixel)
    sar_hidden_states = sar_out.last_hidden_state
    if sar_hidden_states.shape[1] == SAR_NUM_PATCHES + 1:
        sar_hidden_states = sar_hidden_states[:, 1:, :]  # drop a leading CLS token
    sar_embeds = sar_projector(sar_hidden_states.to(sar_projector.proj[1].weight.dtype))

    fused_embeds = build_fused_inputs_embeds(
        inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"], sar_embeds,
    )
    sar_attn_pad = torch.ones((1, sar_embeds.shape[1]), dtype=inputs["attention_mask"].dtype, device=device)
    fused_attention_mask = torch.cat([sar_attn_pad, inputs["attention_mask"]], dim=1)

    output_ids = model.generate(
        inputs_embeds=fused_embeds,
        attention_mask=fused_attention_mask,
        max_new_tokens=max_new_tokens,
    )
    # inputs_embeds-based generate() returns only the newly generated tokens,
    # not the prompt — decode directly, no prompt-length slicing needed.
    return processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)


print("✅ Optical-SAR encoder + projector attached and ready "
      "(call run_optical_sar_inference(image_s2, image_s1, question) for this task)")


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
print("Optical-SAR task requires run_optical_sar_inference(...) — set_adapter alone is not enough for it")
print("=" * 60)