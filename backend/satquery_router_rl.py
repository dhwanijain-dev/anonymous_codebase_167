"""
SatQuery A2A routing RL trainer

Action space (6 actions):
    END
    SINGLE_IMAGE_VQA
    GROUNDING_CAPTIONING
    CHANGE_ANALYSIS
    OPTICAL_SAR_ANALYSIS
    GENERAL_AGENT

The router is a short-horizon contextual DQN.  It does not evaluate VLM
answers during training.  Rewards come from the ground-truth workflow route
stored in a JSONL manifest.

State = projected text embedding + projected image embedding + metadata
        + selected-specialist mask + step fraction.

Default text encoder:
    sentence-transformers/all-MiniLM-L6-v2 (384d)

Default image encoder:
    torchvision MobileNetV3-Small (fast, 2.54M params / 0.06 GFLOPs)

Optional image encoder:
    facebook/dinov2-small (stronger generic visual features, slower)

Manifest example:
{
  "id": "rsvqa_001",
  "query": "What is visible in the image?",
  "images": ["/data/rsvqa_001.jpg"],
  "modalities": ["optical"],
  "required_specialists": ["SINGLE_IMAGE_VQA"],
  "use_general": false
}

General-agent example:
{
  "id": "multi_001",
  "query": "Analyze the change and use optical/SAR evidence together.",
  "images": ["/data/t1.tif", "/data/t2.tif"],
  "modalities": ["optical", "sar"],
  "required_specialists": ["CHANGE_ANALYSIS", "OPTICAL_SAR_ANALYSIS"],
  "use_general": true
}

For a GENERAL_AGENT example, exactly two specialist prerequisites are required.
Without such multi-task route examples, the router cannot learn when the
GENERAL_AGENT action is appropriate from ground truth alone.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

try:
    from transformers import AutoImageProcessor, AutoModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ============================================================
# ACTIONS
# ============================================================

ACTION_END = 0
ACTION_VQA = 1
ACTION_GROUNDING = 2
ACTION_CHANGE = 3
ACTION_OPTICAL_SAR = 4
ACTION_GENERAL = 5
NUM_ACTIONS = 6

ACTION_NAMES = {
    ACTION_END: "END",
    ACTION_VQA: "SINGLE_IMAGE_VQA",
    ACTION_GROUNDING: "GROUNDING_CAPTIONING",
    ACTION_CHANGE: "CHANGE_ANALYSIS",
    ACTION_OPTICAL_SAR: "OPTICAL_SAR_ANALYSIS",
    ACTION_GENERAL: "GENERAL_AGENT",
}

NAME_TO_ACTION = {v: k for k, v in ACTION_NAMES.items()}
SPECIALIST_ACTIONS = {
    ACTION_VQA,
    ACTION_GROUNDING,
    ACTION_CHANGE,
    ACTION_OPTICAL_SAR,
}
SPECIALIST_NAMES = {
    ACTION_NAMES[a] for a in SPECIALIST_ACTIONS
}


# ============================================================
# DATA
# ============================================================

@dataclass
class Sample:
    sample_id: str
    query: str
    images: List[str]
    modalities: List[str]
    required_specialists: List[str]
    use_general: bool = False

    def validate(self):
        if not self.query.strip():
            raise ValueError(f"{self.sample_id}: empty query")
        if not (1 <= len(self.images) <= 2):
            raise ValueError(f"{self.sample_id}: images must contain 1-2 paths")
        if len(self.modalities) != len(self.images):
            raise ValueError(f"{self.sample_id}: modalities length != images length")
        if not self.required_specialists:
            raise ValueError(f"{self.sample_id}: missing required_specialists")
        unknown = set(self.required_specialists) - SPECIALIST_NAMES
        if unknown:
            raise ValueError(f"{self.sample_id}: unknown specialist labels: {unknown}")
        if len(set(self.required_specialists)) != len(self.required_specialists):
            raise ValueError(f"{self.sample_id}: duplicate specialist label")
        if len(self.required_specialists) > 2:
            raise ValueError(f"{self.sample_id}: at most two specialist prerequisites are supported")
        if self.use_general and len(self.required_specialists) != 2:
            raise ValueError(f"{self.sample_id}: GENERAL_AGENT requires exactly two specialists")


def load_manifest(path: str) -> List[Sample]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            s = Sample(
                sample_id=str(row.get("id", f"line_{n}")),
                query=str(row["query"]),
                images=[str(x) for x in row["images"]],
                modalities=[str(x) for x in row["modalities"]],
                required_specialists=[str(x) for x in row["required_specialists"]],
                use_general=bool(row.get("use_general", False)),
            )
            s.validate()
            for image_path in s.images:
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"{s.sample_id}: image not found: {image_path}")
            out.append(s)
    if not out:
        raise ValueError("No samples found in manifest")
    return out


def stratified_split(samples: Sequence[Sample], test_fraction: float, seed: int):
    rng = random.Random(seed)
    groups: Dict[str, List[Sample]] = {}
    for s in samples:
        sig = "+".join(sorted(s.required_specialists))
        if s.use_general:
            sig += "+GENERAL"
        groups.setdefault(sig, []).append(s)

    train, test = [], []
    for group in groups.values():
        rng.shuffle(group)
        if len(group) >= 2:
            n_test = max(1, round(len(group) * test_fraction))
        else:
            n_test = 0
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    if not train:
        raise ValueError("Training split is empty")
    return train, test


# ============================================================
# FROZEN MULTIMODAL FEATURES
# ============================================================

class FrozenFeatureEncoder:
    def __init__(
        self,
        device: str,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_encoder: str = "mobilenet_v3_small",
    ):
        self.device = torch.device(device)
        self.image_encoder_name = image_encoder

        print(f"Loading text encoder: {text_model}")
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
                raise RuntimeError("Install transformers to use DINOv2-Small")
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            self.image_backbone = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device).eval()
            self.image_dim = 384

        else:
            raise ValueError("image_encoder must be mobilenet_v3_small or dinov2_small")

        for p in self.image_backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, queries: Sequence[str]) -> torch.Tensor:
        x = self.text_model.encode(
            list(queries),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return x.float().cpu()

    @torch.no_grad()
    def encode_image_paths(self, paths: Sequence[str]) -> torch.Tensor:
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
            x = self.image_backbone(batch)
            x = self.image_pool(x).flatten(1)
        else:
            x = self.image_backbone(pixel_values=batch).last_hidden_state[:, 0]

        return F.normalize(x.float(), dim=-1).cpu()

    def metadata(self, s: Sample) -> torch.Tensor:
        mods = [m.lower() for m in s.modalities]
        exts = [Path(p).suffix.lower() for p in s.images]
        has_optical = float(any("optical" in m or "multispectral" in m for m in mods))
        has_sar = float(any("sar" in m for m in mods))
        has_temporal = float(any("temporal" in m or "before" in m or "after" in m for m in mods))
        has_tiff = float(any(e in {".tif", ".tiff"} for e in exts))
        has_png_jpg = float(any(e in {".png", ".jpg", ".jpeg"} for e in exts))
        pair = float(len(s.images) == 2)
        optical_sar = float(pair and has_optical and has_sar)
        temporal_pair = float(pair and has_temporal)
        return torch.tensor([
            len(s.images) / 2.0,
            has_optical,
            has_sar,
            has_temporal,
            has_tiff,
            has_png_jpg,
            pair,
            optical_sar,
            temporal_pair,
        ], dtype=torch.float32)

    def encode_samples(self, samples: Sequence[Sample]) -> torch.Tensor:
        print(f"Encoding {len(samples)} queries...")
        text = self.encode_text([s.query for s in samples])

        image_parts = []
        for i, s in enumerate(samples):
            v = self.encode_image_paths(s.images)
            if v.shape[0] == 1:
                mean = v[0]
                diff = torch.zeros_like(mean)
            else:
                mean = v.mean(0)
                diff = torch.abs(v[0] - v[1])
            image_parts.append(torch.cat([mean, diff]))
            if (i + 1) % 100 == 0:
                print(f"  images: {i+1}/{len(samples)}")

        image = torch.stack(image_parts)
        meta = torch.stack([self.metadata(s) for s in samples])
        return torch.cat([text, image, meta], dim=1).float()

    @property
    def raw_dim(self):
        return self.text_dim + 2 * self.image_dim + 9


# ============================================================
# STATE PROJECTION + Q NETWORK
# ============================================================

class StateProjector(nn.Module):
    def __init__(self, text_dim=384, image_dim=576, meta_dim=9, hidden=256):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.meta_dim = meta_dim
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU())
        self.image = nn.Sequential(nn.LayerNorm(2 * image_dim), nn.Linear(2 * image_dim, hidden), nn.GELU())
        self.meta = nn.Sequential(nn.LayerNorm(meta_dim), nn.Linear(meta_dim, 32), nn.GELU())
        self.output_dim = hidden + hidden + 32

    def forward(self, x):
        t = x[:, :self.text_dim]
        a = self.text_dim
        b = a + 2 * self.image_dim
        im = x[:, a:b]
        meta = x[:, b:]
        return torch.cat([self.text(t), self.image(im), self.meta(meta)], dim=-1)


class QNetwork(nn.Module):
    def __init__(self, projected_dim: int, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(projected_dim + 4 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, state):
        return self.net(state)


# ============================================================
# ENVIRONMENT
# ============================================================

class RouteEnv:
    def __init__(self, sample: Sample, base_state: torch.Tensor, max_steps=4):
        self.sample = sample
        self.base_state = base_state.float()
        self.required = {NAME_TO_ACTION[x] for x in sample.required_specialists}
        self.use_general = sample.use_general
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.selected: List[int] = []
        self.steps = 0
        self.done = False
        return self.state()

    def state(self):
        selected_mask = torch.zeros(4)
        idx = {ACTION_VQA: 0, ACTION_GROUNDING: 1, ACTION_CHANGE: 2, ACTION_OPTICAL_SAR: 3}
        for a in self.selected:
            selected_mask[idx[a]] = 1.0
        return torch.cat([
            self.base_state,
            selected_mask,
            torch.tensor([self.steps / max(self.max_steps, 1)], dtype=torch.float32),
        ])

    def allowed(self):
        actions = [ACTION_END]
        actions += [a for a in sorted(SPECIALIST_ACTIONS) if a not in self.selected]
        if len(self.selected) == 2:
            actions.append(ACTION_GENERAL)
        return actions

    def allowed_mask(self):
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        for action in self.allowed():
            mask[action] = 1.0
        return mask

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called after terminal state")
        if action not in self.allowed():
            self.done = True
            return self.state(), -5.0, True, {"invalid": True}

        info: Dict[str, Any] = {}
        reward = 0.0

        if action in SPECIALIST_ACTIONS:
            self.selected.append(action)
            if action in self.required:
                reward = 1.0
                info["correct_specialist"] = True
            else:
                reward = -0.75
                info["irrelevant_specialist"] = True
            self.steps += 1

        elif action == ACTION_GENERAL:
            selected_set = set(self.selected)
            if self.use_general and selected_set == self.required and len(self.selected) == 2:
                reward = 3.0
                info["correct_general"] = True
            elif self.use_general:
                reward = -2.0
                info["wrong_general_context"] = True
            else:
                reward = -1.5
                info["general_not_required"] = True
            self.steps += 1
            self.done = True

        elif action == ACTION_END:
            correct = set(self.selected) == self.required
            if correct and not self.use_general:
                reward = 5.0
                info["correct_end"] = True
            elif correct and self.use_general:
                reward = -4.0
                info["ended_before_general"] = True
            else:
                reward = -5.0
                info["wrong_end"] = True
            self.steps += 1
            self.done = True

        if self.steps >= self.max_steps and not self.done:
            reward -= 5.0
            self.done = True
            info["timeout"] = True

        return self.state(), reward, self.done, info


# ============================================================
# REPLAY + DQN
# ============================================================

@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    next_allowed: torch.Tensor
    done: bool


class Replay:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def add(self, t):
        self.buf.append(t)

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


class DQN:
    def __init__(self, state_dim, device, lr=2e-4, gamma=0.95):
        self.device = torch.device(device)
        self.policy = QNetwork(state_dim).to(self.device)
        self.target = QNetwork(state_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.replay = Replay()
        self.gamma = gamma
        self.steps = 0

    def epsilon(self, start=1.0, end=0.05, decay=5000):
        p = min(self.steps / decay, 1.0)
        return start + p * (end - start)

    def act(self, state, allowed, train=True):
        if train and random.random() < self.epsilon():
            return random.choice(list(allowed))
        with torch.no_grad():
            q = self.policy(state.to(self.device).unsqueeze(0))[0]
        masked = torch.full_like(q, -1e9)
        for a in allowed:
            masked[a] = q[a]
        return int(masked.argmax().item())

    def optimize(self, batch_size=64, target_update=100):
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size)
        s = torch.stack([x.state for x in batch]).to(self.device)
        a = torch.tensor([x.action for x in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        ns = torch.stack([x.next_state for x in batch]).to(self.device)
        d = torch.tensor([x.done for x in batch], dtype=torch.float32, device=self.device)
        next_allowed = torch.stack([x.next_allowed for x in batch]).to(self.device)

        q = self.policy(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            # Double-DQN style selection with legal-action masking.
            next_online = self.policy(ns)
            illegal = next_allowed <= 0
            next_online = next_online.masked_fill(illegal, -1e9)
            next_actions = next_online.argmax(1)
            next_target = self.target(ns).gather(1, next_actions[:, None]).squeeze(1)
            y = r + self.gamma * (1.0 - d) * next_target

        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        self.steps += 1
        if self.steps % target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())


# ============================================================
# ROUTE EVALUATION
# ============================================================

def rollout(policy: QNetwork, sample: Sample, base_state: torch.Tensor, device: str, max_steps=4):
    env = RouteEnv(sample, base_state, max_steps=max_steps)
    state = env.reset()
    trace = []
    while not env.done:
        with torch.no_grad():
            q = policy(state.to(device).unsqueeze(0))[0]
        masked = torch.full_like(q, -1e9)
        for a in env.allowed():
            masked[a] = q[a]
        action = int(masked.argmax().item())
        trace.append(ACTION_NAMES[action])
        state, _, _, _ = env.step(action)
    return trace


def evaluate(policy, samples, features, indices, device):
    exact = 0
    workflow = 0
    details = []
    for local_i, sample in enumerate(samples):
        pred = rollout(policy, sample, features[local_i], device)
        expected_specialists = set(sample.required_specialists)
        predicted_specialists = {x for x in pred if x in SPECIALIST_NAMES}
        pred_general = "GENERAL_AGENT" in pred
        expected_general = sample.use_general
        route_ok = (predicted_specialists == expected_specialists and pred_general == expected_general)
        expected = sorted(sample.required_specialists, key=lambda x: NAME_TO_ACTION[x])
        if sample.use_general:
            expected += ["GENERAL_AGENT"]
        expected += ["END"]
        if pred == expected:
            exact += 1
        if route_ok:
            workflow += 1
        details.append({"id": sample.sample_id, "expected": expected, "predicted": pred})
    n = max(len(samples), 1)
    return {
        "exact_sequence_accuracy": exact / n,
        "workflow_accuracy": workflow / n,
        "details": details,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", default="satquery_router.pt")
    ap.add_argument("--feature-cache", default="satquery_router_features.pt")
    ap.add_argument("--image-encoder", choices=["mobilenet_v3_small", "dinov2_small"], default="mobilenet_v3_small")
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--episodes-per-sample", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = load_manifest(args.manifest)
    train_samples, test_samples = stratified_split(samples, args.test_fraction, args.seed)

    print("Total samples:", len(samples))
    print("Train:", len(train_samples), "Test:", len(test_samples))
    print("Route distribution:", Counter(
        (tuple(sorted(s.required_specialists)), s.use_general) for s in samples
    ))

    encoder = FrozenFeatureEncoder(
        device=args.device,
        text_model=args.text_model,
        image_encoder=args.image_encoder,
    )

    cache_path = Path(args.feature_cache)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("sample_ids") != [s.sample_id for s in samples]:
            raise ValueError("Existing feature cache does not match this manifest")
        raw_features = cached["features"]
        print("Loaded feature cache:", cache_path)
    else:
        raw_features = encoder.encode_samples(samples)
        torch.save({
            "sample_ids": [s.sample_id for s in samples],
            "features": raw_features,
            "text_model": args.text_model,
            "image_encoder": args.image_encoder,
        }, cache_path)
        print("Saved feature cache:", cache_path)

    id_to_i = {s.sample_id: i for i, s in enumerate(samples)}
    train_raw = torch.stack([raw_features[id_to_i[s.sample_id]] for s in train_samples])
    test_raw = torch.stack([raw_features[id_to_i[s.sample_id]] for s in test_samples])

    projector = StateProjector(image_dim=encoder.image_dim).to(args.device)
    policy = DQN(projected_dim=projector.output_dim, device=args.device)
    projector_opt = torch.optim.AdamW(projector.parameters(), lr=5e-4, weight_decay=1e-4)

    # --------------------------------------------
    # Small supervised warm-start from route labels.
    # This stabilizes DQN because the state space is small but the route
    # environment is sequential.
    # --------------------------------------------
    print("Supervised warm-start...")
    for epoch in range(10):
        projected = projector(train_raw.to(args.device))
        states = []
        targets = []
        for i, sample in enumerate(train_samples):
            states.append(torch.cat([projected[i], torch.zeros(4, device=args.device), torch.zeros(1, device=args.device)]))
            first = sorted(sample.required_specialists, key=lambda x: NAME_TO_ACTION[x])[0]
            targets.append(NAME_TO_ACTION[first])
        states = torch.stack(states)
        targets = torch.tensor(targets, dtype=torch.long, device=args.device)
        logits = policy.policy(states)
        loss = F.cross_entropy(logits, targets)
        policy.opt.zero_grad(set_to_none=True)
        projector_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.policy.parameters(), 1.0)
        nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        policy.opt.step()
        projector_opt.step()
        if epoch % 2 == 0:
            print(f"  warmup {epoch:02d} loss={loss.item():.4f}")

    with torch.no_grad():
        train_projected = projector(train_raw.to(args.device)).cpu()
        test_projected = projector(test_raw.to(args.device)).cpu()

    print("Starting DQN...")
    for epoch in range(args.epochs):
        order = list(range(len(train_samples)))
        random.shuffle(order)
        rewards = []
        losses = []

        for i in order:
            sample = train_samples[i]
            for _ in range(args.episodes_per_sample):
                env = RouteEnv(sample, train_projected[i])
                state = env.reset()
                while not env.done:
                    action = policy.act(state, env.allowed(), train=True)
                    next_state, reward, done, _ = env.step(action)
                    policy.replay.add(Transition(
                        state.detach(),
                        action,
                        reward,
                        next_state.detach(),
                        env.allowed_mask(),
                        done,
                    ))
                    state = next_state
                    rewards.append(reward)
                    loss = policy.optimize(args.batch_size)
                    if loss is not None:
                        losses.append(loss)

        metrics = evaluate(policy.policy, test_samples, test_projected, None, args.device)
        print(
            f"epoch {epoch+1:03d}/{args.epochs:03d} | "
            f"eps={policy.epsilon():.3f} | "
            f"reward={np.mean(rewards) if rewards else 0:.3f} | "
            f"loss={np.mean(losses) if losses else 0:.4f} | "
            f"workflow_acc={metrics['workflow_accuracy']:.3f} | "
            f"exact_seq={metrics['exact_sequence_accuracy']:.3f}"
        )

    final = evaluate(policy.policy, test_samples, test_projected, None, args.device)
    print("\nFinal workflow accuracy:", round(final["workflow_accuracy"], 4))
    print("Final exact sequence accuracy:", round(final["exact_sequence_accuracy"], 4))
    for row in final["details"][:10]:
        print("\n", row)

    torch.save({
        "version": 1,
        "action_names": ACTION_NAMES,
        "num_actions": NUM_ACTIONS,
        "text_model": args.text_model,
        "image_encoder": args.image_encoder,
        "image_dim": encoder.image_dim,
        "projector_state_dict": projector.state_dict(),
        "policy_state_dict": policy.policy.state_dict(),
        "projected_dim": projector.output_dim,
        "metrics": final,
    }, args.output)

    print("\nSaved router checkpoint:", args.output)
# FASTAPI INFERENCE WRAPPER
# ============================================================

class SatQueryRouter:
    """
    Load the trained router and produce a specialist/general-agent route.

    This class intentionally does not call any VLM. It only decides which
    agent(s) should be called. The FastAPI orchestration layer can then map
    the returned action names to the actual VLMs/general agent.
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = FrozenFeatureEncoder(
            device=self.device,
            text_model=ckpt["text_model"],
            image_encoder=ckpt["image_encoder"],
        )

        self.projector = StateProjector(
            text_dim=ckpt["text_dim"],
            image_dim=ckpt["image_dim"],
            meta_dim=ckpt["meta_dim"],
        ).to(self.device)
        self.projector.load_state_dict(ckpt["projector_state_dict"])
        self.projector.eval()

        self.policy = QNetwork(
            projected_dim=ckpt["projected_dim"]
        ).to(self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()

    @torch.no_grad()
    def route(
        self,
        query: str,
        image_paths: List[str],
        modalities: List[str],
        max_steps: int = 4,
    ) -> List[Dict[str, Any]]:
        sample = Sample(
            sample_id="inference",
            query=query,
            images=image_paths,
            modalities=modalities,
            # No ground truth is used at inference.
            required_specialists=["SINGLE_IMAGE_VQA"],
            use_general=False,
        )
        sample.validate()

        raw = self.encoder.encode_samples([sample])[0]
        projected = self.projector(raw.unsqueeze(0).to(self.device))[0].cpu()

        selected: List[int] = []
        trace: List[Dict[str, Any]] = []

        for step in range(max_steps):
            selected_mask = torch.zeros(4)
            idx_map = {ACTION_VQA: 0, ACTION_GROUNDING: 1, ACTION_CHANGE: 2, ACTION_OPTICAL_SAR: 3}
            for action in selected:
                if action in idx_map:
                    selected_mask[idx_map[action]] = 1.0

            state = torch.cat([
                projected,
                selected_mask,
                torch.tensor([step / max(max_steps, 1)], dtype=torch.float32),
            ]).to(self.device)

            with torch.no_grad():
                q = self.policy(state.unsqueeze(0))[0]

            allowed = [ACTION_END]
            allowed += [a for a in sorted(SPECIALIST_ACTIONS) if a not in selected]
            if len(selected) == 2:
                allowed.append(ACTION_GENERAL)

            masked = torch.full_like(q, -1e9)
            for action in allowed:
                masked[action] = q[action]

            action = int(masked.argmax().item())
            name = ACTION_NAMES[action]
            trace.append({
                "step": step,
                "action_id": action,
                "action": name,
                "q_value": float(q[action].item()),
            })

            if action in {ACTION_END, ACTION_GENERAL}:
                break

            selected.append(action)

        return trace


if __name__ == "__main__":
    main()
