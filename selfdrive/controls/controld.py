#!/usr/bin/env python3
"""
Control process for drone racing.

Subscribes to dronePlan and droneState, runs PID controllers to convert
desired velocities into roll/pitch/yaw/throttle commands, and publishes
droneControl at 100Hz.

PID axes:
  - Velocity error (world) -> body frame rotation -> pitch/roll commands (deg)
  - Vertical velocity error -> throttle delta (added to gravity feedforward)
  - Yaw error -> yaw rate command (deg/s)
  - When plan is stale (>0.5s), PIDs reset and drone hovers
"""
import math
import time

import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper
from selfdrive.controls.gravity_compensator import (
  quat_to_euler,
  compute_gravity_compensation,
  MASS,
  MAX_THRUST,
)

STALE_PLAN_TIMEOUT = 0.5  # seconds


class PIDController:
  def __init__(self, kp, ki, kd, integral_limit=1.0):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.integral_limit = integral_limit

    self._integral = 0.0
    self._prev_error = 0.0
    self._initialized = False

  def update(self, error, dt):
    if not self._initialized:
      self._prev_error = error
      self._initialized = True

    self._integral += error * dt
    # Anti-windup clamp
    self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

    derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
    self._prev_error = error

    return self.kp * error + self.ki * self._integral + self.kd * derivative

  def reset(self):
    self._integral = 0.0
    self._prev_error = 0.0
    self._initialized = False


def main():
  sm = messaging.SubMaster(['dronePlan', 'droneState'])
  pm = messaging.PubMaster(['droneControl'])

  rk = Ratekeeper(100, print_delay_threshold=None)
  dt = 1.0 / 100.0

  # PID controllers for each axis
  pid_vx = PIDController(kp=5.0, ki=0.5, kd=0.3, integral_limit=10.0)
  pid_vy = PIDController(kp=5.0, ki=0.5, kd=0.3, integral_limit=10.0)
  pid_vz = PIDController(kp=8.0, ki=1.0, kd=0.5, integral_limit=5.0)
  pid_yaw = PIDController(kp=3.0, ki=0.1, kd=0.2, integral_limit=5.0)

  last_plan_time = 0.0
  has_drone_state = False
  has_plan = False

  while True:
    sm.update(0)

    now = time.monotonic()

    if sm.updated['droneState']:
      has_drone_state = True

    # Track plan freshness
    if sm.updated['dronePlan']:
      last_plan_time = now
      has_plan = True

    plan_stale = (now - last_plan_time) > STALE_PLAN_TIMEOUT and last_plan_time > 0
    plan_ok = has_plan and not plan_stale

    if rk.frame % 500 == 0:
      mode = "TRACK" if plan_ok else "HOVER"
      print(f"[controld] frame={rk.frame} mode={mode} has_plan={has_plan} "
            f"has_ds={has_drone_state} plan_stale={plan_stale}")

    # Get current drone state
    if has_drone_state:
      ds = sm['droneState']
      attitude = list(ds.attitude)
      velocity = list(ds.velocity)
      armed = ds.armed

      if len(attitude) == 4:
        roll, pitch, yaw = quat_to_euler(attitude)
      else:
        roll, pitch, yaw = 0.0, 0.0, 0.0

      if len(velocity) == 3:
        vx, vy, vz = velocity
      else:
        vx, vy, vz = 0.0, 0.0, 0.0
    else:
      roll, pitch, yaw = 0.0, 0.0, 0.0
      vx, vy, vz = 0.0, 0.0, 0.0
      armed = False

    # Gravity compensation feedforward
    feedforward, _ = compute_gravity_compensation(roll, pitch)

    msg = messaging.new_message('droneControl')
    dc = msg.droneControl
    dc.armed = armed
    dc.maneuverType = 'stabilized'
    dc.feedforwardThrust = feedforward
    dc.timestamp = int(now * 1e6)

    if not plan_ok:
      # No valid plan — hover with gravity compensation only
      pid_vx.reset()
      pid_vy.reset()
      pid_vz.reset()
      pid_yaw.reset()

      dc.throttle = max(0.0, min(1.0, feedforward)) if armed else 0.0
      dc.roll = 0.0
      dc.pitch = 0.0
      dc.yaw = 0.0
    else:
      dp = sm['dronePlan']
      desired_vel = list(dp.desiredVelocity)
      desired_yaw = dp.desiredYaw

      if len(desired_vel) == 3:
        des_vx, des_vy, des_vz = desired_vel
      else:
        des_vx, des_vy, des_vz = 0.0, 0.0, 0.0

      # Velocity errors in world frame
      err_vx = des_vx - vx
      err_vy = des_vy - vy
      err_vz = des_vz - vz

      # PID outputs in world frame
      cmd_world_x = pid_vx.update(err_vx, dt)
      cmd_world_y = pid_vy.update(err_vy, dt)

      # Rotate world-frame commands to body frame for pitch/roll
      cos_yaw = math.cos(yaw)
      sin_yaw = math.sin(yaw)
      # Body forward = world X rotated by -yaw
      cmd_forward = cmd_world_x * cos_yaw + cmd_world_y * sin_yaw
      cmd_lateral = -cmd_world_x * sin_yaw + cmd_world_y * cos_yaw

      # Forward command -> pitch (nose DOWN = negative pitch for forward flight)
      pitch_cmd = max(-30.0, min(30.0, -cmd_forward))
      # Lateral command -> roll (right roll = positive roll command for rightward motion)
      roll_cmd = max(-30.0, min(30.0, cmd_lateral))

      # Vertical velocity -> throttle delta
      vz_cmd = pid_vz.update(err_vz, dt)
      throttle_delta = vz_cmd / MAX_THRUST
      throttle = feedforward + throttle_delta
      throttle = max(0.0, min(1.0, throttle)) if armed else 0.0

      # Yaw error -> yaw rate command
      yaw_error = desired_yaw - yaw
      yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
      yaw_cmd = pid_yaw.update(yaw_error, dt)
      yaw_cmd = max(-90.0, min(90.0, math.degrees(yaw_cmd)))

      dc.throttle = throttle
      dc.pitch = pitch_cmd
      dc.roll = roll_cmd
      dc.yaw = yaw_cmd

    pm.send('droneControl', msg)

    rk.keep_time()


if __name__ == "__main__":
  main()
