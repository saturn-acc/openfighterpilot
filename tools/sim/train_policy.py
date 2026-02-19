#!/usr/bin/env python3
"""
Train a SAC policy for drone racing using the PyBullet gym environment.

SAC (Soft Actor-Critic) is an off-policy algorithm that's more sample-efficient
than PPO for continuous control. It uses entropy regularization to encourage
exploration and a replay buffer to reuse past experience.

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

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from tools.sim.gym_drone import DroneEnv, GATE_W, GATE_H

MODEL_DIR = "selfdrive/controls/models"

# Curriculum stages: (gate_w, gate_h, min_reward_to_advance)
# Start with big easy gates, shrink as policy learns to thread them
CURRICULUM_STAGES = [
  (4.0, 4.0, 100.0),    # Stage 0: huge gates — learn basic navigation
  (3.0, 3.0, 120.0),    # Stage 1: large gates — refine approach
  (2.25, 2.25, 100.0),  # Stage 2: medium gates — learn precision
  (GATE_W, GATE_H, None),  # Stage 3: real size (1.5m) — final polish
]


class CurriculumCallback(BaseCallback):
  """Shrink gates when eval reward exceeds the stage threshold.

  Checks the rolling mean reward every `check_freq` steps. When it exceeds
  the threshold for the current stage for `patience` consecutive checks,
  advances to the next stage (smaller gates). Both the train and eval envs
  are updated simultaneously.
  """

  def __init__(self, train_env, eval_env, check_freq=20_000, patience=3, verbose=1):
    super().__init__(verbose)
    self.train_env = train_env
    self.eval_env = eval_env
    self.check_freq = check_freq
    self.patience = patience
    self.stage = 0
    self.above_count = 0
    self.recent_rewards = []

    # Apply initial stage
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
    # Collect episode rewards from the monitor
    if len(self.model.ep_info_buffer) > 0:
      self.recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer]

    if self.num_timesteps % self.check_freq != 0:
      return True

    # Check if we should advance
    threshold = CURRICULUM_STAGES[self.stage][2]
    if threshold is None:
      return True  # final stage, no advancing

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

  # SAC uses separate networks for actor (pi) and critic (qf)
  policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

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

  callbacks = []

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
