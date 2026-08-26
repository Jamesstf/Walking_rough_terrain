"""Deterministic 204-D perceptive actor loaded from an RSL-RL checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PerceptiveCheckpointWalkingPolicy:
    """141-D proprioception plus a 63-D height map to 12 joint actions."""

    def __init__(self, checkpoint_path: str, device: str = "auto", clip_actions: float = 100.0):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load the perceptive checkpoint") from exc

        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.clip_actions = float(clip_actions)

        class PerceptiveActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proprio_input = torch.nn.Linear(141, 512)
                self.terrain_encoder = torch.nn.Sequential(
                    torch.nn.Linear(63, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 32),
                    torch.nn.ELU(),
                )
                self.terrain_adapter = torch.nn.Linear(32, 512)
                self.activation_0 = torch.nn.ELU()
                self.proprio_hidden = torch.nn.Linear(512, 128)
                self.activation_1 = torch.nn.ELU()
                self.action_head = torch.nn.Linear(128, 12)

            def forward(self, observation):
                proprioception, terrain = torch.split(observation, (141, 63), dim=-1)
                terrain_latent = self.terrain_encoder(terrain)
                fused = self.proprio_input(proprioception) + self.terrain_adapter(terrain_latent)
                hidden = self.activation_0(fused)
                hidden = self.activation_1(self.proprio_hidden(hidden))
                return self.action_head(hidden)

        self.actor = PerceptiveActor().to(self.device)
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
            raise ValueError(f"Perceptive actor mismatch: missing={missing}, extra={extra}")
        self.actor.load_state_dict(actor_state)
        self.actor.eval()
        self.iteration = checkpoint.get("iter")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (204,):
            raise ValueError(f"Perceptive observation is {observation.shape}, expected (204,)")
        tensor = self.torch.from_numpy(observation).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.actor(tensor).squeeze(0).cpu().numpy()
        return np.clip(output.astype(np.float32), -self.clip_actions, self.clip_actions)

