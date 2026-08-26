"""Load the deterministic actor directly from an RSL-RL training checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CheckpointWalkingPolicy:
    """The 141 -> 512 -> 128 -> 12 actor used by g1_walking_rough."""

    def __init__(self, checkpoint_path: str, device: str = "auto", clip_actions: float = 100.0):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load the rough walking checkpoint") from exc

        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.clip_actions = float(clip_actions)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(141, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 12),
        ).to(self.device)

        checkpoint = torch.load(str(path), map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        actor_state = {
            key[len("actor."):]: value
            for key, value in state.items()
            if key.startswith("actor.")
        }
        expected = set(self.actor.state_dict())
        if set(actor_state) != expected:
            missing = sorted(expected - set(actor_state))
            extra = sorted(set(actor_state) - expected)
            raise ValueError(f"Actor parameters do not match: missing={missing}, extra={extra}")
        self.actor.load_state_dict(actor_state)
        self.actor.eval()
        self.iteration = checkpoint.get("iter")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(np.asarray(observation, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.actor(tensor).squeeze(0).cpu().numpy()
        return np.clip(output.astype(np.float32), -self.clip_actions, self.clip_actions)

