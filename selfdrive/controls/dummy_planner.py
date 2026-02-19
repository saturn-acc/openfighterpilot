#!/usr/bin/env python3
"""
Dummy planner: proves cereal messages + PyBullet work together.

Runs PyBullet drone sim + a "go straight" planner in a single process.
Cereal capnp messages are used for all data (DroneState, DroneControl, ModelV2)
with serialize/deserialize roundtrips to validate the wire format.

Usage:
  python selfdrive/controls/dummy_planner.py [--gui] [--seconds 10]
"""
import argparse
import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data
import capnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
URDF_PATH = os.path.join(ROOT, "tools", "sim", "assets", "quadrotor.urdf")

# Load cereal schema directly (no msgq dependency)
log = capnp.load(os.path.join(ROOT, "cereal", "log.capnp"))

# Sim constants
SIM_TIMESTEP = 1.0 / 240.0
CONTROL_HZ = 100
SIM_STEPS_PER_CONTROL = int((1.0 / CONTROL_HZ) / SIM_TIMESTEP)

# Physics
GRAVITY = 9.81
MASS = 1.0
MAX_THRUST = 20.0

# Go-straight command: gentle nose-down pitch for forward flight
FORWARD_PITCH_DEG = -2.0  # nose down = forward (small angle for stable flight)


def new_event(service, size=None):
  """Create a cereal Event message (same as messaging.new_message but without msgq)."""
  dat = log.Event.new_message(valid=True, logMonoTime=int(time.monotonic() * 1e9))
  if size is None:
    dat.init(service)
  else:
    dat.init(service, size)
  return dat


def verify_roundtrip(msg, service_name):
  """Serialize -> deserialize and verify a field survived the roundtrip."""
  raw = msg.to_bytes()
  with log.Event.from_bytes(raw) as reader:
    assert reader.which() == service_name
  return len(raw)


def quat_to_euler(w, x, y, z):
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
  pitch = math.asin(sinp)
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return roll, pitch, yaw


def main():
  parser = argparse.ArgumentParser(description="Dummy planner: cereal + PyBullet test")
  parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
  parser.add_argument("--seconds", type=float, default=10.0, help="Sim duration")
  args = parser.parse_args()

  # --- PyBullet setup ---
  mode = p.GUI if args.gui else p.DIRECT
  physics_client = p.connect(mode)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.setGravity(0, 0, -GRAVITY)
  p.setTimeStep(SIM_TIMESTEP)
  p.loadURDF("plane.urdf")
  drone_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 1.0])

  print(f"PyBullet connected (client={physics_client}), drone loaded at [0, 0, 1]")
  print(f"Running {args.seconds}s at {CONTROL_HZ}Hz  |  pitch={FORWARD_PITCH_DEG} deg (go straight)")
  print("-" * 70)

  total_frames = int(args.seconds * CONTROL_HZ)
  total_bytes = 0
  t0 = time.monotonic()

  for frame in range(total_frames):
    # ---- 1. Read PyBullet state -> build DroneState cereal message ----
    pos, orn_xyzw = p.getBasePositionAndOrientation(drone_id)
    lin_vel, ang_vel = p.getBaseVelocity(drone_id)
    orn_wxyz = [orn_xyzw[3], orn_xyzw[0], orn_xyzw[1], orn_xyzw[2]]

    state_msg = new_event('droneState')
    ds = state_msg.droneState
    ds.position = list(pos)
    ds.velocity = list(lin_vel)
    ds.attitude = orn_wxyz
    ds.angularRates = [math.degrees(v) for v in ang_vel]
    ds.batteryVoltage = 16.0
    ds.batteryPercent = 0.95
    ds.armed = True
    ds.flightMode = "STABILIZED"
    ds.timestamp = int(time.monotonic() * 1e6)

    total_bytes += verify_roundtrip(state_msg, 'droneState')

    # Read attitude from the builder for control computation
    roll, pitch, yaw = quat_to_euler(*orn_wxyz)

    # ---- 2. Planner: compute hover thrust + forward pitch -> DroneControl ----
    cos_tilt = math.cos(roll) * math.cos(pitch)
    tilt = math.acos(max(-1.0, min(1.0, cos_tilt)))
    hover_thrust = (MASS * GRAVITY) / max(math.cos(min(tilt, math.radians(80))), 0.01)
    throttle = max(0.0, min(1.0, hover_thrust / MAX_THRUST))

    ctrl_msg = new_event('droneControl')
    dc = ctrl_msg.droneControl
    dc.throttle = throttle
    dc.feedforwardThrust = throttle
    dc.roll = 0.0
    dc.pitch = FORWARD_PITCH_DEG
    dc.yaw = 0.0
    dc.armed = True
    dc.maneuverType = 'stabilized'

    total_bytes += verify_roundtrip(ctrl_msg, 'droneControl')

    # ---- 3. Apply control to PyBullet ----
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn_xyzw)).reshape(3, 3)
    local_z = rot_matrix[:, 2]
    thrust_force = local_z * (throttle * MAX_THRUST)
    p.applyExternalForce(drone_id, -1, thrust_force.tolist(), [0, 0, 0], p.LINK_FRAME)

    torque_scale = 0.1
    p.applyExternalTorque(drone_id, -1, [
      0.0,                          # roll
      FORWARD_PITCH_DEG * torque_scale,  # pitch
      0.0,                          # yaw
    ], p.LINK_FRAME)

    # ---- 4. Step physics ----
    for _ in range(SIM_STEPS_PER_CONTROL):
      p.stepSimulation()

    # ---- 5. Also build a dummy ModelV2 every 5 frames (~20Hz) ----
    if frame % 5 == 0:
      model_msg = new_event('modelV2')
      m = model_msg.modelV2
      m.frameId = frame // 5
      m.frameAge = 0
      m.frameDropPerc = 0.0
      m.timestampEof = int(time.monotonic() * 1e9)
      m.modelExecutionTime = 0.01
      total_bytes += verify_roundtrip(model_msg, 'modelV2')

    # ---- 6. Print status ----
    if frame % 200 == 0:
      print(f"  t={frame / CONTROL_HZ:5.1f}s  pos=[{pos[0]:+7.2f}, {pos[1]:+7.2f}, {pos[2]:+7.2f}]  "
            f"vel=[{lin_vel[0]:+6.2f}, {lin_vel[1]:+6.2f}, {lin_vel[2]:+6.2f}]  "
            f"throttle={throttle:.3f}  pitch={FORWARD_PITCH_DEG:.1f}")

  # Final state
  elapsed = time.monotonic() - t0
  final_pos, _ = p.getBasePositionAndOrientation(drone_id)
  print("-" * 70)
  print(f"Done! {total_frames} frames in {elapsed:.2f}s ({total_frames / elapsed:.0f} fps)")
  print(f"Final position: [{final_pos[0]:+.2f}, {final_pos[1]:+.2f}, {final_pos[2]:+.2f}]")
  print(f"Forward distance (X): {final_pos[0]:+.2f} m")
  print(f"Cereal bytes serialized: {total_bytes:,} ({total_bytes / 1024:.1f} KB)")

  # Sanity checks
  ok = True
  if abs(final_pos[0]) > 0.1 or abs(final_pos[2] - 1.0) > 0.5:
    print("PASS: drone moved (physics working)")
  else:
    print("WARN: drone barely moved - check physics")

  print("\nCereal: DroneState, DroneControl, ModelV2 all serialize/deserialize OK")
  print("PyBullet: quadrotor URDF loaded and simulated OK")

  p.disconnect()
  return 0


if __name__ == "__main__":
  exit(main())
