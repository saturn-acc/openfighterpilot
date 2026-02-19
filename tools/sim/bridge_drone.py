#!/usr/bin/env python3
"""
PyBullet Drone Simulation Bridge for OpenFighterPilot.

Standalone bridge connecting a PyBullet quadrotor simulation to openpilot's
messaging system (cereal/msgq). This does NOT extend the car SimulatorBridge.

Architecture:
  PyBullet World <-> bridge_drone.py <-> cereal msgq <-> gravity_compensator.py

Usage:
  python tools/sim/bridge_drone.py [--gui] [--mass 1.0] [--max-thrust 20.0]
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "assets", "quadrotor.urdf")

# Simulation constants
SIM_TIMESTEP = 1.0 / 240.0  # PyBullet default
CONTROL_HZ = 100
SIM_STEPS_PER_CONTROL = int((1.0 / CONTROL_HZ) / SIM_TIMESTEP)  # steps per control iteration


def setup_pybullet(gui=False):
  """Initialize PyBullet physics engine and load quadrotor."""
  mode = p.GUI if gui else p.DIRECT
  physics_client = p.connect(mode)

  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.setGravity(0, 0, -9.81)
  p.setTimeStep(SIM_TIMESTEP)

  # Load ground plane
  p.loadURDF("plane.urdf")

  # Load quadrotor at 1m altitude
  drone_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 1.0], baseOrientation=[0, 0, 0, 1])

  return physics_client, drone_id


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

  p.applyExternalTorque(drone_id, -1, [roll_torque, pitch_torque, yaw_torque], p.LINK_FRAME)


def read_state(drone_id):
  """Read drone state from PyBullet and return as dict."""
  pos, orn_xyzw = p.getBasePositionAndOrientation(drone_id)
  lin_vel, ang_vel = p.getBaseVelocity(drone_id)

  # PyBullet quaternion is [x, y, z, w], convert to [w, x, y, z]
  orn_wxyz = [orn_xyzw[3], orn_xyzw[0], orn_xyzw[1], orn_xyzw[2]]

  # Convert angular velocity from rad/s to deg/s
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

  ds.batteryVoltage = battery_pct * 16.8  # ~4S LiPo
  ds.batteryPercent = battery_pct
  ds.timestamp = int(time.monotonic() * 1e6)
  ds.armed = armed
  ds.flightMode = flight_mode

  pm.send('droneState', msg)


def main():
  parser = argparse.ArgumentParser(description="PyBullet Drone Bridge for OpenFighterPilot")
  parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI visualization")
  parser.add_argument("--mass", type=float, default=1.0, help="Quadrotor mass in kg")
  parser.add_argument("--max-thrust", type=float, default=20.0, help="Maximum thrust in N")
  args = parser.parse_args()

  print(f"[drone_bridge] Starting PyBullet drone bridge (mass={args.mass}kg, max_thrust={args.max_thrust}N)")

  # Setup PyBullet
  physics_client, drone_id = setup_pybullet(gui=args.gui)

  # Setup cereal messaging
  pm = messaging.PubMaster(['droneState'])
  sm = messaging.SubMaster(['droneControl'])

  rk = Ratekeeper(CONTROL_HZ, print_delay_threshold=None)

  armed = True
  battery_pct = 1.0
  battery_drain_rate = 0.0001  # per second at full throttle
  has_control = False
  last_dc = None
  hover_thrust = args.mass * 9.81  # default hover before controller is online

  print("[drone_bridge] Bridge running at 100Hz. Publishing droneState, subscribing droneControl.")

  try:
    while True:
      # Read latest control commands
      sm.update(0)

      if sm.updated['droneControl']:
        last_dc = sm['droneControl']
        armed = last_dc.armed
        has_control = True

        # Drain battery proportional to throttle
        battery_pct = max(0.0, battery_pct - last_dc.throttle * battery_drain_rate / CONTROL_HZ)

      # Step physics — re-apply forces each step (PyBullet clears them after each step)
      for _ in range(SIM_STEPS_PER_CONTROL):
        if has_control:
          apply_control(drone_id, last_dc, args.mass, args.max_thrust)
        else:
          # No controller online yet — hover thrust so drone doesn't freefall
          p.applyExternalForce(drone_id, -1, [0, 0, hover_thrust], [0, 0, 0], p.LINK_FRAME)
        p.stepSimulation()

      # Read state and publish
      state = read_state(drone_id)
      flight_mode = "STABILIZED"
      if sm.updated['droneControl'] and sm['droneControl'].maneuverType == 'acro':
        flight_mode = "ACRO"

      publish_drone_state(pm, state, armed, battery_pct, flight_mode)

      # Print status periodically
      if rk.frame % 500 == 0:
        pos = state['position']
        vel = state['velocity']
        print(f"[drone_bridge] frame={rk.frame} pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] "
              f"vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] armed={armed} bat={battery_pct:.1%}")

      rk.keep_time()

  except KeyboardInterrupt:
    print("\n[drone_bridge] Shutting down...")
  finally:
    p.disconnect()


if __name__ == "__main__":
  main()
