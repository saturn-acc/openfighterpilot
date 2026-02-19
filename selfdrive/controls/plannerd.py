#!/usr/bin/env python3
"""
Gate-seeking planner for drone racing.

Two modes:
  --rl          : RL policy network outputs droneControl directly (replaces controld)
  (default)     : Proportional planner outputs dronePlan for controld

Runs at 20Hz.
"""
import argparse
import math
import time

import numpy as np

import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from selfdrive.controls.gravity_compensator import quat_to_euler

APPROACH_SPEED = 2.0    # m/s max horizontal approach speed
HORIZ_GAIN = 0.5        # proportional gain for horizontal velocity (ramps speed down near gate)
VERT_GAIN = 1.5         # proportional gain for vertical velocity
VERT_MAX_SPEED = 1.0    # m/s max vertical speed
MIN_DISTANCE = 0.5      # m — stop threshold for horizontal approach

# RL action mapping (must match gym_drone.py)
RL_MAX_ANGLE_DEG = 30.0
RL_MAX_YAW_RATE_DEG = 90.0


def main_proportional():
  """Original proportional gate-seeking planner."""
  sm = messaging.SubMaster(['droneState', 'gateDetection'])
  pm = messaging.PubMaster(['dronePlan'])

  rk = Ratekeeper(20, print_delay_threshold=None)
  pub_count = 0

  while True:
    sm.update(0)

    if not sm.updated['droneState']:
      if rk.frame % 40 == 0:
        print("[plannerd] waiting for droneState...")
      rk.keep_time()
      continue

    ds = sm['droneState']
    pos = list(ds.position)

    # Default: hold position (zero velocity, no yaw change)
    vx, vy, vz = 0.0, 0.0, 0.0
    desired_yaw = 0.0

    # Use last-known gate data (not just freshly-updated this cycle)
    gates = sm['gateDetection']
    best_gate = None
    if len(gates) > 0:
      best_gate = max(gates, key=lambda g: g.confidence)

    if best_gate is not None and best_gate.confidence > 0.0:
      dx, dy, dz = list(best_gate.gatePosition)
      horiz_dist = math.sqrt(dx * dx + dy * dy)

      if horiz_dist > MIN_DISTANCE:
        # Normalize horizontal direction and scale by proportional gain
        speed = min(horiz_dist * HORIZ_GAIN, APPROACH_SPEED)
        vx = (dx / horiz_dist) * speed
        vy = (dy / horiz_dist) * speed

        # Only update yaw while actively seeking (avoids atan2 flip at gate)
        desired_yaw = math.atan2(dy, dx)
      # else: within stop threshold, zero horizontal velocity, keep current yaw

      # Vertical: proportional with clamp
      vz = max(-VERT_MAX_SPEED, min(dz * VERT_GAIN, VERT_MAX_SPEED))

    msg = messaging.new_message('dronePlan')
    dp = msg.dronePlan
    dp.desiredPosition = pos
    dp.desiredVelocity = [vx, vy, vz]
    dp.desiredYaw = desired_yaw
    dp.timestamp = int(time.monotonic() * 1e6)

    pm.send('dronePlan', msg)
    pub_count += 1

    if pub_count <= 5 or pub_count % 20 == 0:
      gate_str = f"gate=[{vx:.2f}, {vy:.2f}, {vz:.2f}]" if best_gate else "no gate"
      print(f"[plannerd] #{pub_count} vel=[{vx:.2f}, {vy:.2f}, {vz:.2f}] yaw={desired_yaw:.2f} ({gate_str})")

    rk.keep_time()


def main_rl():
  """RL policy mode — loads trained model and publishes droneControl directly.

  Observation must match gym_drone.py (15 elements):
    [0:4]  gate_dx, gate_dy, gate_dz, gate_yaw_err  (current target)
    [4:7]  vx, vy, vz
    [7:9]  roll, pitch
    [9:13] next_dx, next_dy, next_dz, next_yaw_err  (next gate after current)
    [13]   progress  (gates_passed / total_gates)
    [14]   speed     (scalar)
  """
  from stable_baselines3 import SAC

  model_path = "selfdrive/controls/models/policy_v1"
  print(f"[plannerd-rl] Loading model from {model_path}...")
  model = SAC.load(model_path)
  print("[plannerd-rl] Model loaded.")

  sm = messaging.SubMaster(['droneState', 'gateDetection'])
  pm = messaging.PubMaster(['droneControl'])
  rk = Ratekeeper(20, print_delay_threshold=None)
  pub_count = 0

  # Multi-gate tracking state
  current_gate_idx = 0
  gates_passed = 0
  prev_gate_dist = None  # to detect passage by distance collapse

  while True:
    sm.update(0)

    if not sm.updated['droneState']:
      if rk.frame % 40 == 0:
        print("[plannerd-rl] waiting for droneState...")
      rk.keep_time()
      continue

    ds = sm['droneState']
    attitude = list(ds.attitude)
    roll, pitch, yaw = quat_to_euler(attitude) if len(attitude) == 4 else (0.0, 0.0, 0.0)
    vel = list(ds.velocity)
    speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

    # Get all gates from gateDetection
    gates = list(sm['gateDetection'])
    total_gates = len(gates)

    # Sort gates by gateId so we can index them in order
    gates_by_id = sorted(gates, key=lambda g: g.gateId) if gates else []

    # Simple gate-passage detection: if current target gate distance < 1.0m
    # and was previously > 1.0m, count it as passed
    if current_gate_idx < len(gates_by_id):
      g = gates_by_id[current_gate_idx]
      gp = list(g.gatePosition)
      dist = math.sqrt(gp[0]**2 + gp[1]**2 + gp[2]**2)
      if prev_gate_dist is not None and prev_gate_dist > 1.5 and dist < 1.5:
        gates_passed += 1
        current_gate_idx += 1
        print(f"[plannerd-rl] Gate {current_gate_idx} passed! ({gates_passed}/{total_gates})")
      prev_gate_dist = dist

    # Current target gate
    gate_dx, gate_dy, gate_dz = 50.0, 0.0, 0.0
    if current_gate_idx < len(gates_by_id):
      gp = list(gates_by_id[current_gate_idx].gatePosition)
      gate_dx, gate_dy, gate_dz = gp[0], gp[1], gp[2]
    elif len(gates_by_id) > 0:
      # All gates passed or out of range — target the last known gate
      gp = list(gates_by_id[-1].gatePosition)
      gate_dx, gate_dy, gate_dz = gp[0], gp[1], gp[2]

    gate_yaw_err = math.atan2(gate_dy, gate_dx) - yaw

    # Next gate after current (for turn anticipation)
    next_idx = current_gate_idx + 1
    next_dx, next_dy, next_dz, next_yaw_err = 0.0, 0.0, 0.0, 0.0
    if next_idx < len(gates_by_id):
      np_ = list(gates_by_id[next_idx].gatePosition)
      next_dx, next_dy, next_dz = np_[0], np_[1], np_[2]
      next_yaw_err = math.atan2(next_dy, next_dx) - yaw

    progress = gates_passed / total_gates if total_gates > 0 else 0.0

    # Build observation matching gym_drone.py (no noise at runtime)
    obs = np.array([
      gate_dx, gate_dy, gate_dz, gate_yaw_err,
      vel[0], vel[1], vel[2],
      roll, pitch,
      next_dx, next_dy, next_dz, next_yaw_err,
      progress, speed,
    ], dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)

    # Map actions to physical commands (same mapping as gym_drone.py)
    roll_cmd = float(action[0]) * RL_MAX_ANGLE_DEG
    pitch_cmd = float(action[1]) * RL_MAX_ANGLE_DEG
    yaw_cmd = float(action[2]) * RL_MAX_YAW_RATE_DEG
    throttle_cmd = (float(action[3]) + 1.0) / 2.0  # [-1,1] -> [0,1]

    msg = messaging.new_message('droneControl')
    dc = msg.droneControl
    dc.roll = roll_cmd
    dc.pitch = pitch_cmd
    dc.yaw = yaw_cmd
    dc.throttle = max(0.0, min(1.0, throttle_cmd))
    dc.armed = True
    dc.maneuverType = 'stabilized'

    pm.send('droneControl', msg)
    pub_count += 1

    if pub_count <= 5 or pub_count % 20 == 0:
      print(f"[plannerd-rl] #{pub_count} roll={roll_cmd:.1f} pitch={pitch_cmd:.1f} "
            f"yaw={yaw_cmd:.1f} thr={throttle_cmd:.2f} gate=[{gate_dx:.1f},{gate_dy:.1f},{gate_dz:.1f}] "
            f"passed={gates_passed}/{total_gates}")

    rk.keep_time()


def main():
  parser = argparse.ArgumentParser(description="Drone racing planner")
  parser.add_argument("--rl", action="store_true", help="Use RL policy (outputs droneControl directly)")
  args = parser.parse_args()

  if args.rl:
    main_rl()
  else:
    main_proportional()


if __name__ == "__main__":
  main()
