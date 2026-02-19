#!/usr/bin/env python3
"""
Gravity Compensator Controller for drone flight.

Subscribes to droneState, computes tilt-compensated thrust to counteract gravity,
and publishes droneControl with feedforward thrust.

The core idea: when a quadrotor tilts, the vertical component of its thrust decreases.
To maintain altitude, thrust must increase by 1/cos(tilt_angle).
"""
import math
import time

import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper

# Physical constants
GRAVITY = 9.81       # m/s^2
MASS = 1.0           # kg (default quadrotor mass)
MAX_THRUST = 20.0    # N (default max thrust)
MAX_TILT_RAD = math.radians(80)  # safety clamp to prevent runaway thrust
STALE_TIMEOUT = 0.5  # seconds before declaring droneState stale


def quat_to_euler(q):
  """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians."""
  w, x, y, z = q
  # Roll (x-axis rotation)
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)

  # Pitch (y-axis rotation)
  sinp = 2.0 * (w * y - z * x)
  sinp = max(-1.0, min(1.0, sinp))
  pitch = math.asin(sinp)

  # Yaw (z-axis rotation)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)

  return roll, pitch, yaw


def compute_gravity_compensation(roll, pitch, mass=MASS, max_thrust=MAX_THRUST):
  """
  Compute the throttle needed to counteract gravity given current tilt.

  Returns:
    feedforward_thrust: normalized 0-1 thrust value
    tilt_angle: the effective tilt angle in radians
  """
  # Effective tilt angle from roll and pitch
  cos_tilt = math.cos(roll) * math.cos(pitch)
  tilt_angle = math.acos(max(-1.0, min(1.0, cos_tilt)))

  # Clamp tilt to prevent division by near-zero
  tilt_clamped = min(tilt_angle, MAX_TILT_RAD)
  cos_tilt_clamped = math.cos(tilt_clamped)

  # Required thrust to hover at this tilt
  compensated_thrust = (mass * GRAVITY) / cos_tilt_clamped

  # Normalize to 0-1 range
  feedforward = compensated_thrust / max_thrust
  feedforward = max(0.0, min(1.0, feedforward))

  return feedforward, tilt_angle


def main():
  sm = messaging.SubMaster(['droneState'])
  pm = messaging.PubMaster(['droneControl'])

  rk = Ratekeeper(100, print_delay_threshold=None)

  armed = False
  last_state_time = 0.0

  while True:
    sm.update(0)

    now = time.monotonic()

    # Check if we have fresh droneState
    if sm.updated['droneState']:
      last_state_time = now
      ds = sm['droneState']

      # Extract attitude quaternion
      attitude = list(ds.attitude)
      if len(attitude) == 4:
        roll, pitch, yaw = quat_to_euler(attitude)
      else:
        roll, pitch, yaw = 0.0, 0.0, 0.0

      armed = ds.armed
    else:
      roll, pitch, yaw = 0.0, 0.0, 0.0

    # Safety: disarm if droneState is stale
    if (now - last_state_time) > STALE_TIMEOUT and last_state_time > 0:
      armed = False

    # Compute gravity compensation
    feedforward, tilt_angle = compute_gravity_compensation(roll, pitch)

    # Build droneControl message
    msg = messaging.new_message('droneControl')
    dc = msg.droneControl

    if armed:
      # Hover throttle is the feedforward thrust from gravity compensation
      dc.throttle = max(0.0, min(1.0, feedforward))
      dc.feedforwardThrust = feedforward
    else:
      dc.throttle = 0.0
      dc.feedforwardThrust = 0.0

    # Pass through zero for roll/pitch/yaw (a higher-level planner would set these)
    dc.roll = 0.0
    dc.pitch = 0.0
    dc.yaw = 0.0
    dc.armed = armed
    dc.maneuverType = 'stabilized'

    pm.send('droneControl', msg)

    rk.keep_time()


if __name__ == "__main__":
  main()
