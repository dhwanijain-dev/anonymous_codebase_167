"""
satquery_classifier.py  —  Multi-label deep-learning router for SatQuery
=========================================================================

Architecture
------------
Frozen feature encoder (same as the old RL trainer):
    • Text  : sentence-transformers/all-MiniLM-L6-v2  (384-d, L2-norm)
    • Image : MobileNetV3-Small features (576-d per image, L2-norm)
              → for paired images: [mean, |diff|]  (1152-d)
    • Meta  : 9 handcrafted scalars (modality, pair, tiff, …)

Trainable head:
    StateProjector  →  ClassificationHead
        Linear(text_dim, 256) + GELU
        Linear(2*image_dim, 256) + GELU
        Linear(meta_dim, 32) + GELU
        concat → Linear(544, 128) + GELU + Dropout(0.3)
            → Linear(128, 4)   # raw logits per class
    Loss: BCEWithLogitsLoss  (sigmoid applied during inference)

Labels (4 independent binary outputs):
    0  SINGLE_IMAGE_VQA
    1  GROUNDING_CAPTIONING
    2  CHANGE_ANALYSIS
    3  OPTICAL_SAR_ANALYSIS

Manifest format (same as router_manifest_combined.jsonl):
    {
      "id": "...",
      "query": "...",
      "images": ["rel/path1.png", "rel/path2.png"],
      "modalities": ["optical", "optical"],
      "required_specialists": ["CHANGE_ANALYSIS"]
    }

Usage
-----
    # Train
    python satquery_classifier.py --manifest router_manifest_combined.jsonl

    # With options
    python satquery_classifier.py \\
        --manifest router_manifest_combined.jsonl \\
        --output satquery_clf.pt \\
        --feature-cache satquery_clf_features.pt \\
        --epochs 30 --lr 1e-3 --batch-size 64 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

try:
    from transformers import AutoImageProcessor, AutoModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ============================================================
# LABEL SCHEMA
# ============================================================

CLASSES = [
    "SINGLE_IMAGE_VQA",
    "GROUNDING_CAPTIONING",
    "CHANGE_ANALYSIS",
    "OPTICAL_SAR_ANALYSIS",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


# ============================================================
# DATA
# ============================================================

@dataclass
class Sample:
    sample_id: str
    query: str
    images: List[str]           # absolute or relative paths
    modalities: List[str]
    required_specialists: List[str]

    def label_vector(self) -> torch.Tensor:
        """Return a 4-d binary float tensor."""
        vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for spec in self.required_specialists:
            if spec in CLASS_TO_IDX:
                vec[CLASS_TO_IDX[spec]] = 1.0
        return vec


def load_manifest(path: str, image_root: str | None = None) -> List[Sample]:
    """
    Load samples from a JSONL manifest.

    image_root: optional directory prepended to relative image paths.
                Defaults to the manifest's parent directory.
    """
    manifest_path = Path(path)
    root = Path(image_root) if image_root else manifest_path.parent
    samples: List[Sample] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)

            # Resolve image paths
            raw_images: List[str] = [str(x) for x in row["images"]]
            resolved_images: List[str] = []
            for img in raw_images:
                p = Path(img)
                if not p.is_absolute():
                    p = root / img
                if not p.exists():
                    raise FileNotFoundError(
                        f"Line {line_no}: image not found: {p}"
                    )
                resolved_images.append(str(p))

            specs = [str(x) for x in row["required_specialists"]]
            unknown = set(specs) - set(CLASSES)
            if unknown:
                raise ValueError(
                    f"Line {line_no}: unknown specialists: {unknown}. "
                    f"Valid: {CLASSES}"
                )

            samples.append(Sample(
                sample_id=str(row.get("id", f"line_{line_no}")),
                query=str(row["query"]),
                images=resolved_images,
                modalities=[str(x) for x in row["modalities"]],
                required_specialists=specs,
            ))

    if not samples:
        raise ValueError("No samples found in manifest.")
    return samples


def stratified_split(
    samples: Sequence[Sample],
    test_fraction: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    """Stratify by the frozen label-set string."""
    rng = random.Random(seed)
    groups: Dict[str, List[Sample]] = {}
    for s in samples:
        key = "+".join(sorted(s.required_specialists))
        groups.setdefault(key, []).append(s)

    train, test = [], []
    for group in groups.values():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction)) if len(group) >= 2 else 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    if not train:
        raise ValueError("Training split is empty.")
    return train, test


# ============================================================
# FROZEN FEATURE ENCODER  (shared with old RL file, kept identical)
# ============================================================

class FrozenFeatureEncoder:
    """
    Encode (query, images, modalities) → flat float tensor.

    Output layout per sample:
        [text_emb (384) | image_mean (576) | image_diff (576) | meta (9)]
        = 1545 dimensions
    """

    META_DIM = 9

    def __init__(
        self,
        device: str = "cpu",
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_encoder: str = "mobilenet_v3_small",
    ):
        self.device = torch.device(device)
        self.image_encoder_name = image_encoder

        print(f"[encoder] Loading text model: {text_model}")
        self.text_model = SentenceTransformer(text_model, device=device)
        self.text_dim = 384

        if image_encoder == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT
            net = mobilenet_v3_small(weights=weights)
            self.image_transform = weights.transforms()
            self.image_backbone = net.features.to(self.device).eval()
            self.image_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
            self.image_dim = 576
        elif image_encoder == "dinov2_small":
            if not HAS_TRANSFORMERS:
                raise RuntimeError("Install transformers to use DINOv2-Small.")
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.image_backbone = (
                AutoModel.from_pretrained("facebook/dinov2-small")
                .to(self.device)
                .eval()
            )
            self.image_dim = 384
        else:
            raise ValueError("image_encoder must be mobilenet_v3_small or dinov2_small")

        for p in self.image_backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _encode_text(self, queries: Sequence[str]) -> torch.Tensor:
        return self.text_model.encode(
            list(queries),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).float().cpu()

    @torch.no_grad()
    def _encode_images(self, paths: Sequence[str]) -> torch.Tensor:
        tensors = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.image_encoder_name == "mobilenet_v3_small":
                tensors.append(self.image_transform(img))
            else:
                tensors.append(
                    self.image_processor(images=img, return_tensors="pt")["pixel_values"][0]
                )
        batch = torch.stack(tensors).to(self.device)

        if self.image_encoder_name == "mobilenet_v3_small":
            feats = self.image_backbone(batch)
            feats = self.image_pool(feats).flatten(1)
        else:
            feats = self.image_backbone(pixel_values=batch).last_hidden_state[:, 0]

        return F.normalize(feats.float(), dim=-1).cpu()

    def _metadata(self, s: Sample) -> torch.Tensor:
        mods = [m.lower() for m in s.modalities]
        exts = [Path(p).suffix.lower() for p in s.images]
        has_optical  = float(any("optical" in m or "multispectral" in m for m in mods))
        has_sar      = float(any("sar" in m for m in mods))
        has_temporal = float(any("temporal" in m or "before" in m or "after" in m for m in mods))
        has_tiff     = float(any(e in {".tif", ".tiff"} for e in exts))
        has_png_jpg  = float(any(e in {".png", ".jpg", ".jpeg"} for e in exts))
        pair         = float(len(s.images) == 2)
        optical_sar  = float(pair and has_optical and has_sar)
        temporal_pair = float(pair and has_temporal)
        return torch.tensor(
            [len(s.images) / 2.0, has_optical, has_sar, has_temporal,
             has_tiff, has_png_jpg, pair, optical_sar, temporal_pair],
            dtype=torch.float32,
        )

    def encode_samples(self, samples: Sequence[Sample]) -> torch.Tensor:
        """Returns tensor of shape (N, raw_dim)."""
        print(f"[encoder] Encoding {len(samples)} queries …")
        text_embs = self._encode_text([s.query for s in samples])

        image_parts = []
        for i, s in enumerate(samples):
            v = self._encode_images(s.images)
            mean = v.mean(0)
            diff = torch.abs(v[0] - v[1]) if v.shape[0] == 2 else torch.zeros_like(mean)
            image_parts.append(torch.cat([mean, diff]))
            if (i + 1) % 200 == 0:
                print(f"[encoder]   images {i+1}/{len(samples)}")

        image_embs = torch.stack(image_parts)
        meta       = torch.stack([self._metadata(s) for s in samples])
        return torch.cat([text_embs, image_embs, meta], dim=1).float()

    @property
    def raw_dim(self) -> int:
        return self.text_dim + 2 * self.image_dim + self.META_DIM


# ============================================================
# NEURAL-NETWORK MODULES
# ============================================================

class StateProjector(nn.Module):
    """
    Projects the concatenated raw features into a compact embedding.
    Same structure as the old RL StateProjector.
    """

    def __init__(
        self,
        text_dim: int = 384,
        image_dim: int = 576,
        meta_dim: int = 9,
        hidden: int = 256,
    ):
        super().__init__()
        self.text_dim  = text_dim
        self.image_dim = image_dim
        self.meta_dim  = meta_dim

        self.text  = nn.Sequential(nn.LayerNorm(text_dim),       nn.Linear(text_dim,       hidden), nn.GELU())
        self.image = nn.Sequential(nn.LayerNorm(2 * image_dim),  nn.Linear(2 * image_dim,  hidden), nn.GELU())
        self.meta  = nn.Sequential(nn.LayerNorm(meta_dim),       nn.Linear(meta_dim,       32),     nn.GELU())
        self.output_dim = hidden + hidden + 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, : self.text_dim]
        im = x[:, self.text_dim : self.text_dim + 2 * self.image_dim]
        m  = x[:, self.text_dim + 2 * self.image_dim :]
        return torch.cat([self.text(t), self.image(im), self.meta(m)], dim=1)


class ClassificationHead(nn.Module):
    """
    Multi-label classification head.
    Outputs raw logits (apply sigmoid externally or use BCEWithLogitsLoss).
    """

    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)          # raw logits, shape (B, num_classes)


class SatQueryMultiLabelClassifier(nn.Module):
    """
    End-to-end multi-label classifier:
        raw_features → StateProjector → ClassificationHead → logits
    """

    def __init__(
        self,
        text_dim: int = 384,
        image_dim: int = 576,
        meta_dim: int = 9,
        hidden: int = 256,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projector = StateProjector(text_dim, image_dim, meta_dim, hidden)
        self.head      = ClassificationHead(self.projector.output_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.projector(x))   # logits


# ============================================================
# TORCH DATASET
# ============================================================

class ManifestDataset(Dataset):
    """Wraps pre-computed raw features + label tensors."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0]
        self.features = features
        self.labels   = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for feats, labels in loader:
            logits = model(feats.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = (torch.sigmoid(logits) >= threshold).float()

    # Per-class accuracy
    per_class_acc = (preds == labels).float().mean(0)
    # Exact match (all 4 labels correct per sample)
    exact_match   = (preds == labels).all(dim=1).float().mean().item()
    # Hamming accuracy
    hamming_acc   = (preds == labels).float().mean().item()
    # Macro F1 (per-class)
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1_per    = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1  = f1_per.mean().item()

    return {
        "exact_match":   round(exact_match, 4),
        "hamming_acc":   round(hamming_acc, 4),
        "macro_f1":      round(macro_f1, 4),
        "per_class_acc": {CLASSES[i]: round(per_class_acc[i].item(), 4) for i in range(NUM_CLASSES)},
        "per_class_f1":  {CLASSES[i]: round(f1_per[i].item(), 4) for i in range(NUM_CLASSES)},
    }


# ============================================================
# TRAINING ENTRY POINT
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="SatQuery multi-label classifier trainer")
    ap.add_argument("--manifest",      required=True,  help="Path to JSONL manifest (router_manifest_combined.jsonl)")
    ap.add_argument("--image-root",    default=None,   help="Root directory for relative image paths (defaults to manifest parent)")
    ap.add_argument("--output",        default="satquery_clf.pt",          help="Output checkpoint path")
    ap.add_argument("--feature-cache", default="satquery_clf_features.pt", help="Cached raw features")
    ap.add_argument("--image-encoder", choices=["mobilenet_v3_small", "dinov2_small"], default="mobilenet_v3_small")
    ap.add_argument("--text-model",    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs",        type=int,   default=30)
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight-decay",  type=float, default=1e-4)
    ap.add_argument("--dropout",       type=float, default=0.3)
    ap.add_argument("--hidden",        type=int,   default=256)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--threshold",     type=float, default=0.5,  help="Sigmoid threshold for binary decision")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--pos-weight",    type=float, default=None, help="Positive class weight for BCEWithLogitsLoss (handles imbalance)")
    args = ap.parse_args()

    # ── Seeding ──────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # ── Load manifest ─────────────────────────────────────────
    samples = load_manifest(args.manifest, args.image_root)
    train_samples, test_samples = stratified_split(samples, args.test_fraction, args.seed)

    print(f"Total samples : {len(samples)}")
    print(f"Train / Test  : {len(train_samples)} / {len(test_samples)}")

    from collections import Counter
    label_dist = Counter(
        "+".join(sorted(s.required_specialists)) for s in samples
    )
    print("Label distribution:", dict(label_dist))

    # ── Feature cache ─────────────────────────────────────────
    cache_path = Path(args.feature_cache)
    id_order   = [s.sample_id for s in samples]

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("sample_ids") != id_order:
            print("[cache] sample IDs mismatch — recomputing …")
            cache_path.unlink()
        else:
            raw_features = cached["features"]
            print(f"[cache] Loaded from {cache_path}")

    if not cache_path.exists():
        encoder = FrozenFeatureEncoder(
            device=args.device,
            text_model=args.text_model,
            image_encoder=args.image_encoder,
        )
        raw_features = encoder.encode_samples(samples)
        torch.save({
            "sample_ids":    id_order,
            "features":      raw_features,
            "text_model":    args.text_model,
            "image_encoder": args.image_encoder,
        }, cache_path)
        print(f"[cache] Saved to {cache_path}")
        # Release encoder memory
        del encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Build label tensors ───────────────────────────────────
    id_to_idx = {s.sample_id: i for i, s in enumerate(samples)}
    all_labels = torch.stack([s.label_vector() for s in samples])

    train_idx  = [id_to_idx[s.sample_id] for s in train_samples]
    test_idx   = [id_to_idx[s.sample_id] for s in test_samples]
    train_feats  = raw_features[train_idx]
    test_feats   = raw_features[test_idx]
    train_labels = all_labels[train_idx]
    test_labels  = all_labels[test_idx]

    train_ds = ManifestDataset(train_feats, train_labels)
    test_ds  = ManifestDataset(test_feats,  test_labels)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── Model ─────────────────────────────────────────────────
    raw_dim   = raw_features.shape[1]
    text_dim  = 384
    image_dim = 576          # mobilenet_v3_small; adjust if using dinov2
    meta_dim  = raw_dim - text_dim - 2 * image_dim   # should be 9

    model = SatQueryMultiLabelClassifier(
        text_dim=text_dim,
        image_dim=image_dim,
        meta_dim=meta_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    # ── Loss (BCEWithLogitsLoss = sigmoid + BCE, numerically stable) ──
    if args.pos_weight is not None:
        pw = torch.full((NUM_CLASSES,), args.pos_weight, device=device)
    else:
        # Auto positive-class weight from training label frequencies
        pos_freq = train_labels.mean(0).clamp(1e-3, 1 - 1e-3)
        pw = ((1 - pos_freq) / pos_freq).to(device)
        print(f"[loss] Auto pos_weight per class: { {CLASSES[i]: round(pw[i].item(),2) for i in range(NUM_CLASSES)} }")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────
    print("\n── Starting training ──")
    best_f1   = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for feats, labels in train_dl:
            feats  = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            metrics = compute_metrics(model, test_dl, device, args.threshold)
            print(
                f"epoch {epoch:03d}/{args.epochs:03d} | "
                f"loss={epoch_loss/n_batches:.4f} | "
                f"exact={metrics['exact_match']:.4f} | "
                f"hamming={metrics['hamming_acc']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )
            if metrics["macro_f1"] > best_f1:
                best_f1    = metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────
    final_metrics = compute_metrics(model, test_dl, device, args.threshold)
    print("\n── Final test metrics ──")
    print(f"  Exact match  : {final_metrics['exact_match']}")
    print(f"  Hamming acc  : {final_metrics['hamming_acc']}")
    print(f"  Macro F1     : {final_metrics['macro_f1']}")
    print(f"  Per-class F1 : {final_metrics['per_class_f1']}")

    # ── Save checkpoint ───────────────────────────────────────
    torch.save({
        "version":         2,
        "classes":         CLASSES,
        "num_classes":     NUM_CLASSES,
        "text_model":      args.text_model,
        "image_encoder":   args.image_encoder,
        "text_dim":        text_dim,
        "image_dim":       image_dim,
        "meta_dim":        meta_dim,
        "hidden":          args.hidden,
        "dropout":         args.dropout,
        "threshold":       args.threshold,
        "model_state_dict": model.state_dict(),
        "metrics":         final_metrics,
    }, args.output)

    print(f"\nCheckpoint saved → {args.output}")


# ============================================================
# INFERENCE WRAPPER  (used by clf_router.py)
# ============================================================

class SatQueryClassifier:
    """
    Load a trained checkpoint and classify (query, image_paths, modalities)
    into a set of specialist labels.

    Usage
    -----
        clf = SatQueryClassifier("satquery_clf.pt")
        labels = clf.predict(
            query="Did the area change?",
            image_paths=["/tmp/a.png", "/tmp/b.png"],
            modalities=["optical", "optical"],
        )
        # labels → e.g. ["CHANGE_ANALYSIS"]
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.device    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes   = ckpt["classes"]
        self.threshold = ckpt.get("threshold", 0.5)

        self.encoder = FrozenFeatureEncoder(
            device=str(self.device),
            text_model=ckpt["text_model"],
            image_encoder=ckpt["image_encoder"],
        )

        self.model = SatQueryMultiLabelClassifier(
            text_dim=ckpt["text_dim"],
            image_dim=ckpt["image_dim"],
            meta_dim=ckpt["meta_dim"],
            hidden=ckpt["hidden"],
            dropout=0.0,              # no dropout at inference
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        query: str,
        image_paths: List[str],
        modalities: List[str],
        threshold: float | None = None,
    ) -> List[str]:
        """
        Returns the list of predicted specialist class names.
        Falls back to ["SINGLE_IMAGE_VQA"] if nothing crosses the threshold.
        """
        thr = threshold if threshold is not None else self.threshold

        sample = Sample(
            sample_id="inference",
            query=query,
            images=image_paths,
            modalities=modalities,
            required_specialists=["SINGLE_IMAGE_VQA"],  # dummy — not used
        )

        raw = self.encoder.encode_samples([sample])            # (1, raw_dim)
        logits = self.model(raw.to(self.device)).squeeze(0)    # (num_classes,)
        probs  = torch.sigmoid(logits).cpu()

        predicted = [
            self.classes[i]
            for i in range(len(self.classes))
            if probs[i].item() >= thr
        ]
        return predicted if predicted else ["SINGLE_IMAGE_VQA"]

    @torch.no_grad()
    def predict_proba(
        self,
        query: str,
        image_paths: List[str],
        modalities: List[str],
    ) -> Dict[str, float]:
        """Returns per-class sigmoid probabilities."""
        sample = Sample(
            sample_id="inference",
            query=query,
            images=image_paths,
            modalities=modalities,
            required_specialists=["SINGLE_IMAGE_VQA"],
        )
        raw    = self.encoder.encode_samples([sample])
        logits = self.model(raw.to(self.device)).squeeze(0)
        probs  = torch.sigmoid(logits).cpu()
        return {self.classes[i]: round(probs[i].item(), 4) for i in range(len(self.classes))}


if __name__ == "__main__":
    main()
