#!/usr/bin/env python3
"""
Train a SAC policy for drone racing using the PyBullet gym environment.

Uses a custom attention-based feature extractor that groups observations into
semantic tokens (gate, velocity, attitude, next gate, progress) and applies
multi-head self-attention so the network learns to dynamically focus on what
matters — e.g., centering precision near gates vs heading when far away.

Supports curriculum learning: starts with wide gates and shrinks them as the
policy improves, making it much easier to learn gate-threading behavior.

Usage:
  python tools/sim/train_policy.py                       # 5M steps, headless
  python tools/sim/train_policy.py --timesteps 1000      # quick smoke test
  python tools/sim/train_policy.py --gui                  # eval env with GUI
  python tools/sim/train_policy.py --resume               # continue from best_model
  python tools/sim/train_policy.py --curriculum           # curriculum learning (wide→narrow gates)
"""
import argparse
import os
import sys

# Ensure project root is on sys.path when run as a script
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
  sys.path.insert(0, _project_root)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tools.sim.gym_drone import DroneEnv, GATE_W, GATE_H

MODEL_DIR = "selfdrive/controls/models"

# Curriculum stages: (gate_w, gate_h, min_reward_to_advance)
CURRICULUM_STAGES = [
  (4.0, 4.0, 100.0),
  (3.0, 3.0, 120.0),
  (2.25, 2.25, 100.0),
  (GATE_W, GATE_H, None),
]

# Observation token groups (indices into the 15-element obs vector):
#   [0:4]   current gate: dx, dy, dz, yaw_err
#   [4:7]   velocity: vx, vy, vz
#   [7:9]   attitude: roll, pitch
#   [9:13]  next gate: dx, dy, dz, yaw_err
#   [13:15] progress: progress, speed
OBS_TOKEN_SLICES = [(0, 4), (4, 7), (7, 9), (9, 13), (13, 15)]
NUM_TOKENS = len(OBS_TOKEN_SLICES)


class AttentionExtractor(BaseFeaturesExtractor):
  """Groups obs into semantic tokens, projects to embeddings, applies self-attention.

  Observation (15 elements) is split into 5 tokens:
    - Current gate (4): relative position + yaw error
    - Velocity (3): world-frame velocity
    - Attitude (2): roll, pitch
    - Next gate (4): lookahead for turn anticipation
    - Progress (2): course progress + speed

  Each token is projected to a common embedding dim, then multi-head self-attention
  lets the network learn dynamic weighting — e.g., attend more to gate centering when
  close, more to velocity/heading when far.
  """

  def __init__(self, observation_space: gym.spaces.Box, embed_dim: int = 32,
               num_heads: int = 2, num_layers: int = 2):
    # Output features = NUM_TOKENS * embed_dim (flattened after attention)
    features_dim = NUM_TOKENS * embed_dim
    super().__init__(observation_space, features_dim=features_dim)

    self.embed_dim = embed_dim
    self.token_slices = OBS_TOKEN_SLICES

    # Per-token linear projection to common embedding dim
    self.token_projections = nn.ModuleList([
      nn.Sequential(
        nn.Linear(end - start, embed_dim),
        nn.ReLU(),
      )
      for start, end in self.token_slices
    ])

    # Learnable positional encoding (one per token)
    self.pos_encoding = nn.Parameter(torch.randn(1, NUM_TOKENS, embed_dim) * 0.02)

    # Transformer encoder (self-attention layers)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embed_dim,
      nhead=num_heads,
      dim_feedforward=embed_dim * 4,
      dropout=0.0,
      activation="gelu",
      batch_first=True,
    )
    self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # Layer norm on output
    self.out_norm = nn.LayerNorm(features_dim)

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    batch_size = observations.shape[0]

    # Split obs into tokens and project each to embed_dim
    tokens = []
    for i, (start, end) in enumerate(self.token_slices):
      token_input = observations[:, start:end]
      token_embed = self.token_projections[i](token_input)
      tokens.append(token_embed)

    # Stack into (batch, num_tokens, embed_dim) and add positional encoding
    x = torch.stack(tokens, dim=1) + self.pos_encoding

    # Self-attention
    x = self.attention(x)

    # Flatten tokens into single feature vector
    x = x.reshape(batch_size, -1)
    x = self.out_norm(x)
    return x


class CheckpointCallback(BaseCallback):
  """Save the model every N seconds (default: 1 hour)."""

  def __init__(self, save_dir, save_interval_sec=3600, verbose=1):
    super().__init__(verbose)
    self.save_dir = save_dir
    self.save_interval_sec = save_interval_sec
    self.last_save_time = None

  def _on_training_start(self):
    import time
    self.last_save_time = time.time()

  def _on_step(self) -> bool:
    import time
    now = time.time()
    if now - self.last_save_time >= self.save_interval_sec:
      path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
      self.model.save(path)
      self.last_save_time = now
      if self.verbose:
        print(f"\n[checkpoint] Saved at {self.num_timesteps} steps → {path}.zip")
    return True


class CurriculumCallback(BaseCallback):
  """Shrink gates when eval reward exceeds the stage threshold."""

  def __init__(self, train_env, eval_env, check_freq=20_000, patience=3, verbose=1):
    super().__init__(verbose)
    self.train_env = train_env
    self.eval_env = eval_env
    self.check_freq = check_freq
    self.patience = patience
    self.stage = 0
    self.above_count = 0
    self.recent_rewards = []
    self._set_stage(0)

  def _set_stage(self, stage_idx):
    w, h, _ = CURRICULUM_STAGES[stage_idx]
    self.stage = stage_idx
    self.above_count = 0
    self.train_env.set_gate_size(w, h)
    self.eval_env.set_gate_size(w, h)
    if self.verbose:
      print(f"\n[curriculum] Stage {stage_idx}: gate size {w:.1f}x{h:.1f}m")

  def _on_step(self) -> bool:
    if len(self.model.ep_info_buffer) > 0:
      self.recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer]

    if self.num_timesteps % self.check_freq != 0:
      return True

    threshold = CURRICULUM_STAGES[self.stage][2]
    if threshold is None:
      return True

    if len(self.recent_rewards) < 5:
      return True

    mean_rew = np.mean(self.recent_rewards)
    if self.verbose:
      w, h, _ = CURRICULUM_STAGES[self.stage]
      print(f"[curriculum] Step {self.num_timesteps}: mean_reward={mean_rew:.1f}, "
            f"threshold={threshold:.0f}, stage={self.stage} ({w:.1f}x{h:.1f}m), "
            f"streak={self.above_count}/{self.patience}")

    if mean_rew >= threshold:
      self.above_count += 1
      if self.above_count >= self.patience:
        next_stage = self.stage + 1
        if next_stage < len(CURRICULUM_STAGES):
          self._set_stage(next_stage)
          print(f"[curriculum] *** ADVANCED to stage {next_stage}! ***")
    else:
      self.above_count = 0

    return True


def main():
  parser = argparse.ArgumentParser(description="Train SAC drone racing policy")
  parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
  parser.add_argument("--gui", action="store_true", help="Show GUI for eval environment")
  parser.add_argument("--resume", action="store_true", help="Resume training from best_model")
  parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (wide→narrow gates)")
  args = parser.parse_args()

  # Start with wide gates if curriculum, else real size
  if args.curriculum:
    init_w, init_h = CURRICULUM_STAGES[0][0], CURRICULUM_STAGES[0][1]
    print(f"[train] Curriculum learning enabled: starting with {init_w}x{init_h}m gates")
  else:
    init_w, init_h = GATE_W, GATE_H

  print(f"[train] Creating environments (gui={args.gui} for eval)...")
  env = DroneEnv(gui=False, gate_w=init_w, gate_h=init_h)
  eval_env = DroneEnv(gui=args.gui, gate_w=init_w, gate_h=init_h)

  # Custom attention extractor + MLP heads for actor/critic
  policy_kwargs = dict(
    features_extractor_class=AttentionExtractor,
    features_extractor_kwargs=dict(embed_dim=32, num_heads=2, num_layers=2),
    net_arch=dict(pi=[256, 128], qf=[256, 128]),
    share_features_extractor=False,  # actor and critic get their own attention
  )

  if args.resume:
    resume_path = os.path.join(MODEL_DIR, "best_model.zip")
    print(f"[train] Resuming from {resume_path}...")
    model = SAC.load(resume_path, env=env, device="cpu")
  else:
    model = SAC(
      "MlpPolicy",
      env,
      policy_kwargs=policy_kwargs,
      learning_rate=3e-4,
      buffer_size=1_000_000,
      learning_starts=10_000,
      batch_size=256,
      tau=0.005,
      gamma=0.99,
      train_freq=1,
      gradient_steps=1,
      ent_coef="auto",
      verbose=1,
      device="cpu",
    )

  # Print architecture
  total_params = sum(p.numel() for p in model.policy.parameters())
  print(f"[train] Attention-based SAC policy: {total_params:,} parameters")

  callbacks = []

  # Save checkpoint every hour so killing the process doesn't lose progress
  ckpt_cb = CheckpointCallback(MODEL_DIR, save_interval_sec=3600)
  callbacks.append(ckpt_cb)

  eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    eval_freq=10_000,
    n_eval_episodes=10,
  )
  callbacks.append(eval_cb)

  if args.curriculum:
    curr_cb = CurriculumCallback(env, eval_env, check_freq=20_000, patience=3)
    callbacks.append(curr_cb)

  print(f"[train] Starting SAC training for {args.timesteps} timesteps...")
  model.learn(total_timesteps=args.timesteps, callback=callbacks)
  model.save(os.path.join(MODEL_DIR, "policy_v1"))
  print(f"[train] Done. Model saved to {MODEL_DIR}/policy_v1.zip")

  env.close()
  eval_env.close()


if __name__ == "__main__":
  main()
