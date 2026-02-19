#!/usr/bin/env python3
"""
PyBullet Drone Simulation Bridge with Gate Detection.

Enhanced bridge that adds visual gate markers and publishes gateDetection
messages alongside droneState for the full racing stack.

Architecture:
  PyBullet World <-> bridge_pybullet.py <-> cereal msgq <-> plannerd/controld

Usage:
  python tools/sim/bridge_pybullet.py [--gui] [--mass 1.0] [--max-thrust 20.0]
"""
import argparse
import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from selfdrive.controls.gravity_compensator import quat_to_euler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "assets", "quadrotor.urdf")

# Simulation constants
SIM_TIMESTEP = 1.0 / 240.0
CONTROL_HZ = 100
SIM_STEPS_PER_CONTROL = int((1.0 / CONTROL_HZ) / SIM_TIMESTEP)

# Gate defaults
DEFAULT_GATE_POSITIONS = [
  [5.0, 0.0, 1.0],
  [10.0, 2.0, 1.5],
  [15.0, -1.0, 1.0],
]
GATE_WIDTH = 1.5
GATE_HEIGHT = 1.5
GATE_DETECTION_HZ_DIVISOR = 5  # publish gate detection every 5th frame (100/5 = 20Hz)

# FPV camera constants
FPV_WIDTH = 320
FPV_HEIGHT = 240
FPV_FOV = 90
FPV_HZ_DIVISOR = 5  # capture every 5th frame (100/5 = 20Hz)


def setup_pybullet(gui=False):
  """Initialize PyBullet physics engine and load quadrotor."""
  mode = p.GUI if gui else p.DIRECT
  physics_client = p.connect(mode)

  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.setGravity(0, 0, -9.81)
  p.setTimeStep(SIM_TIMESTEP)

  p.loadURDF("plane.urdf")
  drone_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 1.0], baseOrientation=[0, 0, 0, 1])

  return physics_client, drone_id


def spawn_gates(positions=None):
  """Create visual gate markers at the given positions. Returns list of (gate_id, position)."""
  if positions is None:
    positions = DEFAULT_GATE_POSITIONS

  gates = []
  for pos in positions:
    half_extents = [0.05, GATE_WIDTH / 2.0, GATE_HEIGHT / 2.0]
    visual_shape = p.createVisualShape(
      p.GEOM_BOX,
      halfExtents=half_extents,
      rgbaColor=[1.0, 0.5, 0.0, 0.8],  # orange, slightly transparent
    )
    gate_id = p.createMultiBody(
      baseMass=0,
      baseVisualShapeIndex=visual_shape,
      basePosition=pos,
    )
    gates.append((gate_id, list(pos)))
  return gates


def apply_control(drone_id, control_msg, mass, max_thrust):
  """Apply DroneControl commands with inner-loop attitude stabilization.

  control_msg fields:
    throttle (0-1): upward force along drone's body Z-axis, scaled by max_thrust
    roll: desired roll angle (degrees)
    pitch: desired pitch angle (degrees)
    yaw: desired yaw rate (degrees/s)

  An inner PD loop converts desired angles into appropriate torques,
  preventing the instability of raw torque application.
  """
  # Thrust along body Z-axis
  thrust = control_msg.throttle * max_thrust
  p.applyExternalForce(drone_id, -1, [0, 0, thrust], [0, 0, 0], p.LINK_FRAME)

  # Read current attitude and angular velocity for PD control
  _, orn_xyzw = p.getBasePositionAndOrientation(drone_id)
  _, ang_vel = p.getBaseVelocity(drone_id)
  roll, pitch, yaw = p.getEulerFromQuaternion(orn_xyzw)

  # PD gains (tuned for Ixx=Iyy=0.0043, Izz=0.0072)
  ANGLE_KP = 2.0    # Nm per radian of angle error
  ANGLE_KD = 0.2    # Nm per rad/s of angular rate
  YAW_RATE_KP = 0.05  # Nm per rad/s of yaw rate error

  # PD control for roll and pitch angles
  # Note: controld convention (negative pitch = forward) is opposite to
  # PyBullet Euler convention (positive pitch = nose down = forward).
  # Negate both roll and pitch to align conventions.
  desired_roll = -math.radians(control_msg.roll)
  desired_pitch = -math.radians(control_msg.pitch)
  roll_torque = ANGLE_KP * (desired_roll - roll) - ANGLE_KD * ang_vel[0]
  pitch_torque = ANGLE_KP * (desired_pitch - pitch) - ANGLE_KD * ang_vel[1]

  # P control for yaw rate
  desired_yaw_rate = math.radians(control_msg.yaw)
  yaw_torque = YAW_RATE_KP * (desired_yaw_rate - ang_vel[2])

  # Clamp torques to prevent violent oscillations
  MAX_TORQUE = 0.4
  roll_torque = max(-MAX_TORQUE, min(MAX_TORQUE, roll_torque))
  pitch_torque = max(-MAX_TORQUE, min(MAX_TORQUE, pitch_torque))
  yaw_torque = max(-MAX_TORQUE, min(MAX_TORQUE, yaw_torque))

  p.applyExternalTorque(drone_id, -1, [roll_torque, pitch_torque, yaw_torque], p.LINK_FRAME)


def read_state(drone_id):
  """Read drone state from PyBullet and return as dict."""
  pos, orn_xyzw = p.getBasePositionAndOrientation(drone_id)
  lin_vel, ang_vel = p.getBaseVelocity(drone_id)

  # PyBullet quaternion is [x, y, z, w], convert to [w, x, y, z]
  orn_wxyz = [orn_xyzw[3], orn_xyzw[0], orn_xyzw[1], orn_xyzw[2]]
  ang_vel_deg = [math.degrees(v) for v in ang_vel]

  return {
    'position': list(pos),
    'velocity': list(lin_vel),
    'attitude': orn_wxyz,
    'angular_rates': ang_vel_deg,
  }


def publish_drone_state(pm, state, armed, battery_pct, flight_mode):
  """Publish DroneState message to cereal."""
  msg = messaging.new_message('droneState')
  ds = msg.droneState

  ds.position = state['position']
  ds.velocity = state['velocity']
  ds.attitude = state['attitude']
  ds.angularRates = state['angular_rates']

  ds.batteryVoltage = battery_pct * 16.8
  ds.batteryPercent = battery_pct
  ds.timestamp = int(time.monotonic() * 1e6)
  ds.armed = armed
  ds.flightMode = flight_mode

  pm.send('droneState', msg)


def publish_gate_detection(pm, drone_state, gate_positions):
  """Compute relative gate positions from drone and publish gateDetection."""
  drone_pos = np.array(drone_state['position'])

  msg = messaging.new_message('gateDetection', len(gate_positions))

  for i, gate_world_pos in enumerate(gate_positions):
    gate_pos = np.array(gate_world_pos)
    rel_world = gate_pos - drone_pos
    distance = float(np.linalg.norm(rel_world))
    gate_yaw = math.atan2(rel_world[1], rel_world[0])

    gd = msg.gateDetection[i]
    gd.gatePosition = rel_world.tolist()
    gd.gateDimensions = [GATE_WIDTH, GATE_HEIGHT]
    gd.confidence = max(0.0, min(1.0, 1.0 - distance / 50.0))
    gd.gateYaw = gate_yaw
    gd.gateId = i
    gd.distance = distance

  pm.send('gateDetection', msg)


def capture_fpv_image(drone_id):
  """Capture an FPV image from the drone's perspective using PyBullet's camera."""
  pos, orn_xyzw = p.getBasePositionAndOrientation(drone_id)

  # Get rotation matrix from quaternion to find drone's local axes
  rot_matrix = p.getMatrixFromQuaternion(orn_xyzw)
  # Local X-axis (forward): columns of rotation matrix
  forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
  # Local Z-axis (up)
  up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

  # Camera target is along the drone's forward axis
  target = [pos[0] + forward[0], pos[1] + forward[1], pos[2] + forward[2]]

  view_matrix = p.computeViewMatrix(
    cameraEyePosition=pos,
    cameraTargetPosition=target,
    cameraUpVector=up,
  )
  proj_matrix = p.computeProjectionMatrixFOV(
    fov=FPV_FOV,
    aspect=FPV_WIDTH / FPV_HEIGHT,
    nearVal=0.05,
    farVal=100.0,
  )
  _, _, rgba, _, _ = p.getCameraImage(
    width=FPV_WIDTH,
    height=FPV_HEIGHT,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=p.ER_TINY_RENDERER,
  )
  rgb = np.array(rgba, dtype=np.uint8).reshape(FPV_HEIGHT, FPV_WIDTH, 4)[:, :, :3]
  return rgb


def publish_fpv_image(pm, rgb_array, frame_counter):
  """Publish a DroneCameraState message with raw RGB image bytes."""
  msg = messaging.new_message('droneCameraState')
  dcs = msg.droneCameraState
  dcs.frameId = frame_counter
  dcs.timestamp = int(time.monotonic() * 1e6)
  dcs.image = rgb_array.tobytes()
  dcs.width = FPV_WIDTH
  dcs.height = FPV_HEIGHT
  pm.send('droneCameraState', msg)


def main():
  parser = argparse.ArgumentParser(description="PyBullet Drone Bridge with Gates")
  parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI visualization")
  parser.add_argument("--mass", type=float, default=1.0, help="Quadrotor mass in kg")
  parser.add_argument("--max-thrust", type=float, default=20.0, help="Maximum thrust in N")
  parser.add_argument("--gates", type=str, default=None,
                      help="Gate positions as 'x1,y1,z1;x2,y2,z2;...' (default: 3 gates)")
  args = parser.parse_args()

  # Parse gate positions
  if args.gates:
    gate_positions = []
    for g in args.gates.split(';'):
      coords = [float(c) for c in g.strip().split(',')]
      gate_positions.append(coords)
  else:
    gate_positions = [list(g) for g in DEFAULT_GATE_POSITIONS]

  print(f"[bridge_pybullet] Starting (mass={args.mass}kg, max_thrust={args.max_thrust}N)")
  print(f"[bridge_pybullet] {len(gate_positions)} gates:")
  for i, gp in enumerate(gate_positions):
    print(f"  Gate {i}: [{gp[0]:.1f}, {gp[1]:.1f}, {gp[2]:.1f}]")

  physics_client, drone_id = setup_pybullet(gui=args.gui)
  gates = spawn_gates(gate_positions)

  pm = messaging.PubMaster(['droneState', 'gateDetection', 'droneCameraState'])
  sm = messaging.SubMaster(['droneControl'])

  rk = Ratekeeper(CONTROL_HZ, print_delay_threshold=None)

  armed = True
  battery_pct = 1.0
  battery_drain_rate = 0.0001
  has_control = False  # True once we've received the first droneControl
  last_dc = None

  # Hover thrust to apply before controld is online
  hover_thrust = args.mass * 9.81

  print("[bridge_pybullet] Running at 100Hz. Publishing droneState+gateDetection.")

  try:
    while True:
      sm.update(0)

      if sm.updated['droneControl']:
        has_control = True
        last_dc = sm['droneControl']
        armed = last_dc.armed
        battery_pct = max(0.0, battery_pct - last_dc.throttle * battery_drain_rate / CONTROL_HZ)

      # Re-apply forces each sim step (PyBullet clears them after each step)
      for _ in range(SIM_STEPS_PER_CONTROL):
        if has_control:
          apply_control(drone_id, last_dc, args.mass, args.max_thrust)
        else:
          # No controller online yet — hover thrust so drone doesn't freefall
          p.applyExternalForce(drone_id, -1, [0, 0, hover_thrust], [0, 0, 0], p.LINK_FRAME)
        p.stepSimulation()

      state = read_state(drone_id)
      flight_mode = "STABILIZED"
      if sm.updated['droneControl'] and sm['droneControl'].maneuverType == 'acro':
        flight_mode = "ACRO"

      publish_drone_state(pm, state, armed, battery_pct, flight_mode)

      # FPV chase cam behind drone
      if args.gui and rk.frame % 5 == 0:
        pos = state['position']
        attitude = state['attitude']
        _, _, yaw = quat_to_euler(attitude)
        p.resetDebugVisualizerCamera(
          cameraDistance=0.3,
          cameraYaw=math.degrees(yaw) + 180,
          cameraPitch=-10,
          cameraTargetPosition=pos,
        )

      # Publish gate detection at 20Hz (every 5th frame)
      if rk.frame % GATE_DETECTION_HZ_DIVISOR == 0:
        all_gate_positions = [gp for _, gp in gates]
        publish_gate_detection(pm, state, all_gate_positions)

      # Capture and publish FPV image at 20Hz
      if rk.frame % FPV_HZ_DIVISOR == 0:
        rgb = capture_fpv_image(drone_id)
        publish_fpv_image(pm, rgb, rk.frame // FPV_HZ_DIVISOR)

      if rk.frame % 500 == 0:
        pos = state['position']
        vel = state['velocity']
        print(f"[bridge_pybullet] frame={rk.frame} pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] "
              f"vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] armed={armed} bat={battery_pct:.1%}")

      rk.keep_time()

  except KeyboardInterrupt:
    print("\n[bridge_pybullet] Shutting down...")
  finally:
    p.disconnect()


if __name__ == "__main__":
  main()
