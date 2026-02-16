#!/usr/bin/env python3
"""
Dummy fly-forward planner for drone racing.

Subscribes to droneState only (no vision input). Commands a constant
forward velocity rotated by current yaw into world frame, with
proportional altitude hold.

Runs at 20Hz.
"""
import math
import time

import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from selfdrive.controls.gravity_compensator import quat_to_euler

FORWARD_SPEED = 2.0  # m/s
TARGET_ALT = 1.0     # meters
ALT_GAIN = 1.0       # proportional gain for altitude hold


def main():
  sm = messaging.SubMaster(['droneState'])
  pm = messaging.PubMaster(['dronePlan'])

  rk = Ratekeeper(20, print_delay_threshold=None)
  pub_count = 0

  while True:
    sm.update(0)

    if sm.updated['droneState']:
      ds = sm['droneState']

      attitude = list(ds.attitude)
      if len(attitude) == 4:
        _, _, yaw = quat_to_euler(attitude)
      else:
        yaw = 0.0

      # Forward velocity rotated by current yaw into world frame
      vx_world = FORWARD_SPEED * math.cos(yaw)
      vy_world = FORWARD_SPEED * math.sin(yaw)

      # Altitude hold with proportional gain
      pos = list(ds.position)
      alt_error = TARGET_ALT - pos[2]
      vz_world = alt_error * ALT_GAIN

      msg = messaging.new_message('dronePlan')
      dp = msg.dronePlan
      dp.desiredPosition = pos
      dp.desiredVelocity = [vx_world, vy_world, vz_world]
      dp.desiredYaw = yaw  # maintain current heading
      dp.timestamp = int(time.monotonic() * 1e6)

      pm.send('dronePlan', msg)
      pub_count += 1

      if pub_count <= 5 or pub_count % 20 == 0:
        print(f"[plannerd] #{pub_count} vel=[{vx_world:.2f}, {vy_world:.2f}, {vz_world:.2f}] yaw={yaw:.2f}")

    elif rk.frame % 40 == 0:
      print(f"[plannerd] waiting for droneState...")

    rk.keep_time()


if __name__ == "__main__":
  main()
