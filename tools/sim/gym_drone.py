#!/usr/bin/env python3
"""
Gymnasium environment for drone racing with PyBullet.

Generates random multi-gate courses each episode so the policy learns to
navigate any track layout — turns, altitude changes, varying gate counts —
not just one fixed gate straight ahead.

Observation (15 elements, all relative — no absolute positions):
  [0:3]  gate_dx, gate_dy, gate_dz     — vector to current target gate
  [3]    gate_yaw_err                   — yaw error toward current gate
  [4:7]  vx, vy, vz                    — world-frame velocity
  [7:9]  roll, pitch                    — attitude
  [9:12] next_dx, next_dy, next_dz     — vector to gate AFTER current (for anticipating turns)
  [12]   next_yaw_err                   — yaw error toward next gate
  [13]   progress                       — gates_passed / total_gates
  [14]   speed                          — scalar speed

Usage:
  from tools.sim.gym_drone import DroneEnv
  env = DroneEnv(gui=False)
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action)
"""
import math
import os

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "assets", "quadrotor.urdf")

# Physics constants (must match bridge_pybullet)
SIM_TIMESTEP = 1.0 / 240.0
CONTROL_HZ = 100
SIM_STEPS_PER_CONTROL = int((1.0 / CONTROL_HZ) / SIM_TIMESTEP)  # 2

# Drone constants
MASS = 1.0          # kg
MAX_THRUST = 20.0   # N
GRAVITY = 9.81      # m/s^2

# PD gains (must match bridge_pybullet apply_control)
ANGLE_KP = 2.0
ANGLE_KD = 0.2
YAW_RATE_KP = 0.05
MAX_TORQUE = 0.4    # Nm, clamp torque to prevent violent oscillations

# Action smoothing (exponential moving average)
ACTION_SMOOTH = 0.7  # 0=no smoothing, 1=fully smooth (never changes)

# Action mapping ranges
MAX_ANGLE_DEG = 30.0
MAX_YAW_RATE_DEG = 90.0

# Gate dimensions (final/real size — curriculum may start wider)
GATE_W = 1.5  # meters
GATE_H = 1.5  # meters

# Course generation
MIN_GATES = 3
MAX_GATES = 6
GATE_DIST_RANGE = (3.0, 6.0)     # distance between consecutive gates
GATE_TURN_MAX = math.pi / 4      # max heading change per gate (~45 deg)
GATE_ALT_RANGE = (0.8, 2.5)      # altitude bounds
GATE_ALT_CHANGE_MAX = 0.5        # max altitude change per gate

# Takeoff
START_Z = 0.05          # spawn on the ground
TARGET_ALT = 1.0        # target hover altitude before heading to gate
TAKEOFF_GRACE = 50      # steps before ground collision counts (0.5s to lift off)

# Episode limits
MAX_STEPS = 2000        # enough time for multi-gate courses
BOUNDS_MARGIN = 5.0     # margin beyond course extent before OOB
BOUNDS_Z = 10.0

# Observation noise
OBS_NOISE_STD = 0.05

# Domain randomization ranges (randomized each episode in reset)
# Tightened for initial training — widen after policy learns basics
DR_MASS_RANGE = (0.9, 1.1)
DR_MAX_THRUST_RANGE = (18.0, 22.0)
DR_KP_RANGE = (1.8, 2.2)
DR_KD_RANGE = (0.18, 0.22)
DR_DRAG_RANGE = (0.0, 0.15)
DR_WIND_CONST_MAX = 0.2      # N, constant wind per axis per episode
DR_WIND_GUST_MAX = 0.15      # N, random gust per axis per step
DR_OBS_NOISE_RANGE = (0.03, 0.08)
DR_OBS_BIAS_MAX = 0.05
DR_MOTOR_LAG_RANGE = (0.85, 1.0)  # alpha: 1.0=instant, 0.85=slight lag
DR_LATENCY_RANGE = (0, 2)        # action delay in steps (0 or 1)


class DroneEnv(gym.Env):
  """PyBullet drone racing environment with random multi-gate courses."""

  metadata = {"render_modes": ["human"]}

  def __init__(self, gui=False, gate_w=None, gate_h=None):
    super().__init__()
    self.gui = gui
    self.gate_w = gate_w if gate_w is not None else GATE_W
    self.gate_h = gate_h if gate_h is not None else GATE_H

    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    # Connect PyBullet
    mode = p.GUI if gui else p.DIRECT
    self.physics_client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    self.drone_id = None
    self.gate_bar_ids = []
    self.plane_id = None
    self.step_count = 0
    self.prev_dist = None
    self.prev_pos = None
    self.has_taken_off = False

    # Multi-gate course state
    self.course_gates = []      # list of np.array([x, y, z])
    self.gate_yaws = []         # yaw angle each gate faces (approach direction)
    self.current_gate_idx = 0
    self.gates_passed = 0
    self.total_gates = 0
    self.bounds_x = (-20.0, 20.0)
    self.bounds_y = (-20.0, 20.0)

    # Domain randomization state (set in reset)
    self.ep_mass = MASS
    self.ep_max_thrust = MAX_THRUST
    self.ep_kp = ANGLE_KP
    self.ep_kd = ANGLE_KD
    self.ep_drag = 0.0
    self.ep_wind = np.zeros(3)
    self.ep_obs_noise = OBS_NOISE_STD
    self.ep_obs_bias = np.zeros(15, dtype=np.float32)
    self.ep_motor_lag = 1.0
    self.ep_latency = 0
    self.filtered_thrust = 0.0
    self.action_buffer = []
    self.smooth_action = np.zeros(4, dtype=np.float32)  # smoothed RL output
    self.current_gust = np.zeros(3)  # smoothed wind gust

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    rng = self.np_random

    # Randomize physics parameters for this episode
    self.ep_mass = rng.uniform(*DR_MASS_RANGE)
    self.ep_max_thrust = rng.uniform(*DR_MAX_THRUST_RANGE)
    self.ep_kp = rng.uniform(*DR_KP_RANGE)
    self.ep_kd = rng.uniform(*DR_KD_RANGE)
    self.ep_drag = rng.uniform(*DR_DRAG_RANGE)
    self.ep_wind = rng.uniform(-DR_WIND_CONST_MAX, DR_WIND_CONST_MAX, size=3)
    self.ep_obs_noise = rng.uniform(*DR_OBS_NOISE_RANGE)
    self.ep_obs_bias = rng.uniform(-DR_OBS_BIAS_MAX, DR_OBS_BIAS_MAX, size=15).astype(np.float32)
    self.ep_motor_lag = rng.uniform(*DR_MOTOR_LAG_RANGE)
    self.ep_latency = int(rng.integers(*DR_LATENCY_RANGE))
    self.filtered_thrust = 0.0
    self.action_buffer = []
    self.smooth_action = np.zeros(4, dtype=np.float32)
    self.current_gust = np.zeros(3)

    # Generate random multi-gate course
    self._generate_course()

    p.resetSimulation(physicsClientId=self.physics_client)
    p.setGravity(0, 0, -GRAVITY)
    p.setTimeStep(SIM_TIMESTEP)

    self.plane_id = p.loadURDF("plane.urdf")
    self.drone_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, START_Z], baseOrientation=[0, 0, 0, 1])

    # Apply randomized mass to drone
    p.changeDynamics(self.drone_id, -1, mass=self.ep_mass)

    self._spawn_gates()

    self.step_count = 0
    self.current_gate_idx = 0
    self.gates_passed = 0
    drone_pos, _ = p.getBasePositionAndOrientation(self.drone_id)
    self.prev_dist = self._dist_to_current_gate(drone_pos)
    self.prev_pos = list(drone_pos)
    self.has_taken_off = False

    obs = self._get_obs()
    return obs, {}

  def step(self, action):
    action = np.clip(action, -1.0, 1.0)

    # Smooth action to prevent visual jitter (EMA filter)
    self.smooth_action = ACTION_SMOOTH * self.smooth_action + (1.0 - ACTION_SMOOTH) * action
    sa = self.smooth_action

    # Map smoothed actions to physical commands
    roll_deg = float(sa[0]) * MAX_ANGLE_DEG
    pitch_deg = float(sa[1]) * MAX_ANGLE_DEG
    yaw_rate_deg = float(sa[2]) * MAX_YAW_RATE_DEG
    throttle = (float(sa[3]) + 1.0) / 2.0  # [-1,1] -> [0,1]

    # Handle action latency
    if self.ep_latency > 0:
      self.action_buffer.append((throttle, roll_deg, pitch_deg, yaw_rate_deg))
      if len(self.action_buffer) > self.ep_latency:
        throttle, roll_deg, pitch_deg, yaw_rate_deg = self.action_buffer.pop(0)
      else:
        throttle, roll_deg, pitch_deg, yaw_rate_deg = 0.5, 0.0, 0.0, 0.0

    # Smooth wind gust (updated once per control step, not per substep)
    gust_target = self.np_random.uniform(-DR_WIND_GUST_MAX, DR_WIND_GUST_MAX, size=3)
    self.current_gust = 0.8 * self.current_gust + 0.2 * gust_target
    wind_force = (self.ep_wind + self.current_gust).tolist()

    # Apply control for each substep (forces cleared each stepSimulation)
    for _ in range(SIM_STEPS_PER_CONTROL):
      self._apply_action(throttle, roll_deg, pitch_deg, yaw_rate_deg)
      # Wind: constant + smoothed gust (same force for all substeps)
      p.applyExternalForce(self.drone_id, -1, wind_force, [0, 0, 0], p.WORLD_FRAME)
      # Aerodynamic drag: F_drag = -drag * v
      lin_vel, _ = p.getBaseVelocity(self.drone_id)
      drag_force = [-self.ep_drag * v for v in lin_vel]
      p.applyExternalForce(self.drone_id, -1, drag_force, [0, 0, 0], p.WORLD_FRAME)
      p.stepSimulation()

    self.step_count += 1

    # Read state
    pos, orn_xyzw = p.getBasePositionAndOrientation(self.drone_id)

    # Check gate passage for current target
    gate_just_passed = False
    if self.current_gate_idx < self.total_gates:
      if self._check_gate_passage(pos, self.current_gate_idx):
        gate_just_passed = True
        self.gates_passed += 1
        self.current_gate_idx += 1

    all_gates_passed = self.current_gate_idx >= self.total_gates
    curr_dist = self._dist_to_current_gate(pos) if not all_gates_passed else 0.0

    # Check termination
    collision = self._check_collision(pos)
    out_of_bounds = (pos[0] < self.bounds_x[0] or pos[0] > self.bounds_x[1] or
                     pos[1] < self.bounds_y[0] or pos[1] > self.bounds_y[1] or
                     pos[2] > BOUNDS_Z)
    truncated = self.step_count >= MAX_STEPS

    # Track takeoff
    just_took_off = False
    if not self.has_taken_off and pos[2] >= TARGET_ALT:
      self.has_taken_off = True
      just_took_off = True

    # Compute reward
    # Key insight: shaping rewards must dominate crash penalty so
    # the policy gets useful gradient from partial progress, not just
    # "everything is -50 because I always crash eventually."
    reward = -0.01  # mild time penalty encourages speed

    if collision:
      reward += -20.0
    elif gate_just_passed:
      reward += 50.0
      if all_gates_passed:
        reward += 100.0 * (1.0 - self.step_count / MAX_STEPS)
    else:
      if not self.has_taken_off:
        alt_gain = pos[2] - self.prev_pos[2]
        reward += alt_gain * 15.0
      elif not all_gates_passed:
        if just_took_off:
          reward += 5.0
        # Distance shaping toward current gate
        reward += (self.prev_dist - curr_dist) * 10.0
        # Gate alignment bonus: reward being centered on the gate approach
        # (stronger as drone gets close — teaches precision threading)
        if curr_dist < 3.0:
          gate = self.course_gates[self.current_gate_idx]
          gyaw = self.gate_yaws[self.current_gate_idx]
          # Lateral offset in gate frame
          ny, nx = -math.sin(gyaw), math.cos(gyaw)
          lateral = -(pos[0] - gate[0]) * (-ny) + (pos[1] - gate[1]) * nx
          vertical = pos[2] - gate[2]
          centering = 1.0 - (abs(lateral) + abs(vertical)) / (self.gate_w / 2.0 + self.gate_h / 2.0)
          reward += max(0.0, centering) * 2.0

    terminated = all_gates_passed or collision or out_of_bounds

    # Update prev_dist: if gate just passed, reset to distance to new target
    self.prev_dist = curr_dist
    self.prev_pos = list(pos)

    obs = self._get_obs()
    info = {
      "gates_passed": self.gates_passed,
      "total_gates": self.total_gates,
      "all_gates_passed": all_gates_passed,
      "collision": collision,
      "out_of_bounds": out_of_bounds,
      "has_taken_off": self.has_taken_off,
    }
    return obs, reward, terminated, truncated, info

  def _generate_course(self):
    """Generate a random winding multi-gate course.

    Each gate is placed at a random distance and heading change from the
    previous one, creating courses with turns, altitude changes, and varying
    length. The first gate is biased forward since the drone starts facing +X.
    """
    rng = self.np_random
    n_gates = int(rng.integers(MIN_GATES, MAX_GATES + 1))

    gates = []
    yaws = []
    # First gate heading: mostly forward (drone faces +X)
    heading = float(rng.uniform(-math.pi / 6, math.pi / 6))
    pos = np.array([0.0, 0.0, 1.5])  # reference point near drone start

    for i in range(n_gates):
      dist = rng.uniform(*GATE_DIST_RANGE)
      if i > 0:
        heading += float(rng.uniform(-GATE_TURN_MAX, GATE_TURN_MAX))
      dz = float(rng.uniform(-GATE_ALT_CHANGE_MAX, GATE_ALT_CHANGE_MAX))

      gate_pos = pos + np.array([
        dist * math.cos(heading),
        dist * math.sin(heading),
        dz,
      ])
      gate_pos[2] = np.clip(gate_pos[2], GATE_ALT_RANGE[0], GATE_ALT_RANGE[1])

      # Gate faces the approach direction
      gate_yaw = math.atan2(gate_pos[1] - pos[1], gate_pos[0] - pos[0])

      gates.append(gate_pos.copy())
      yaws.append(float(gate_yaw))
      pos = gate_pos

    self.course_gates = gates
    self.gate_yaws = yaws
    self.total_gates = n_gates

    # Dynamic bounds from course extent
    all_x = [0.0] + [g[0] for g in gates]
    all_y = [0.0] + [g[1] for g in gates]
    self.bounds_x = (min(all_x) - BOUNDS_MARGIN, max(all_x) + BOUNDS_MARGIN)
    self.bounds_y = (min(all_y) - BOUNDS_MARGIN, max(all_y) + BOUNDS_MARGIN)

  def _apply_action(self, throttle, roll_deg, pitch_deg, yaw_rate_deg):
    """PD inner loop with domain randomization."""
    # Motor lag: first-order low-pass filter on thrust
    target_thrust = throttle * self.ep_max_thrust
    self.filtered_thrust = (self.ep_motor_lag * target_thrust +
                            (1.0 - self.ep_motor_lag) * self.filtered_thrust)
    p.applyExternalForce(self.drone_id, -1, [0, 0, self.filtered_thrust], [0, 0, 0], p.LINK_FRAME)

    _, orn_xyzw = p.getBasePositionAndOrientation(self.drone_id)
    _, ang_vel = p.getBaseVelocity(self.drone_id)
    roll, pitch, _yaw = p.getEulerFromQuaternion(orn_xyzw)

    # Negate roll/pitch to match bridge convention
    desired_roll = -math.radians(roll_deg)
    desired_pitch = -math.radians(pitch_deg)
    roll_torque = self.ep_kp * (desired_roll - roll) - self.ep_kd * ang_vel[0]
    pitch_torque = self.ep_kp * (desired_pitch - pitch) - self.ep_kd * ang_vel[1]

    desired_yaw_rate = math.radians(yaw_rate_deg)
    yaw_torque = YAW_RATE_KP * (desired_yaw_rate - ang_vel[2])

    # Clamp torques to prevent violent oscillations
    roll_torque = max(-MAX_TORQUE, min(MAX_TORQUE, roll_torque))
    pitch_torque = max(-MAX_TORQUE, min(MAX_TORQUE, pitch_torque))
    yaw_torque = max(-MAX_TORQUE, min(MAX_TORQUE, yaw_torque))

    p.applyExternalTorque(self.drone_id, -1, [roll_torque, pitch_torque, yaw_torque], p.LINK_FRAME)

  def _spawn_gates(self):
    """Create collision-bar gates for the entire course, oriented by gate yaw."""
    self.gate_bar_ids = []
    bar_color = [1.0, 0.5, 0.0, 1.0]  # orange
    hw = self.gate_w / 2.0
    hh = self.gate_h / 2.0

    for gpos, gyaw in zip(self.course_gates, self.gate_yaws):
      gx, gy, gz = float(gpos[0]), float(gpos[1]), float(gpos[2])
      cy, sy = math.cos(gyaw), math.sin(gyaw)
      orn = p.getQuaternionFromEuler([0, 0, gyaw])

      # Bar positions: top/bottom at gate center ± hh vertically,
      # left/right offset by ±hw along the gate's local Y axis.
      # Local Y in world coords = (-sin(yaw), cos(yaw), 0)
      bars = [
        ([0.05, hw, 0.05], [gx, gy, gz + hh]),                         # top
        ([0.05, hw, 0.05], [gx, gy, gz - hh]),                         # bottom
        ([0.05, 0.05, hh], [gx + sy * hw, gy - cy * hw, gz]),          # left
        ([0.05, 0.05, hh], [gx - sy * hw, gy + cy * hw, gz]),          # right
      ]
      for half_ext, bar_pos in bars:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=bar_color)
        bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                basePosition=bar_pos, baseOrientation=orn)
        self.gate_bar_ids.append(bid)

  def _get_obs(self):
    """Build 15-element observation vector (all relative, no absolute coords)."""
    pos, orn_xyzw = p.getBasePositionAndOrientation(self.drone_id)
    lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn_xyzw)

    # Current target gate
    if self.current_gate_idx < self.total_gates:
      gate = self.course_gates[self.current_gate_idx]
    else:
      gate = self.course_gates[-1]
    gate_dx = gate[0] - pos[0]
    gate_dy = gate[1] - pos[1]
    gate_dz = gate[2] - pos[2]
    gate_yaw_err = math.atan2(gate_dy, gate_dx) - yaw

    # Next gate after current (so policy can anticipate turns)
    next_idx = self.current_gate_idx + 1
    if next_idx < self.total_gates:
      ngate = self.course_gates[next_idx]
      next_dx = ngate[0] - pos[0]
      next_dy = ngate[1] - pos[1]
      next_dz = ngate[2] - pos[2]
      next_yaw_err = math.atan2(next_dy, next_dx) - yaw
    else:
      next_dx, next_dy, next_dz = 0.0, 0.0, 0.0
      next_yaw_err = 0.0

    progress = self.gates_passed / self.total_gates if self.total_gates > 0 else 1.0
    speed = math.sqrt(lin_vel[0]**2 + lin_vel[1]**2 + lin_vel[2]**2)

    obs = np.array([
      gate_dx, gate_dy, gate_dz, gate_yaw_err,
      lin_vel[0], lin_vel[1], lin_vel[2],
      roll, pitch,
      next_dx, next_dy, next_dz, next_yaw_err,
      progress, speed,
    ], dtype=np.float32)

    # Noise + per-episode bias
    obs += obs * self.np_random.normal(0.0, self.ep_obs_noise, size=obs.shape).astype(np.float32)
    obs += self.ep_obs_bias
    return obs

  def _dist_to_current_gate(self, pos):
    """Euclidean distance from drone to current target gate center."""
    if self.current_gate_idx >= self.total_gates:
      return 0.0
    gate = self.course_gates[self.current_gate_idx]
    return math.sqrt((gate[0] - pos[0])**2 + (gate[1] - pos[1])**2 + (gate[2] - pos[2])**2)

  def _check_gate_passage(self, pos, gate_idx):
    """Detect gate passage: drone crosses gate plane from approach side within bounds."""
    gpos = self.course_gates[gate_idx]
    gyaw = self.gate_yaws[gate_idx]

    # Gate normal = approach direction
    nx = math.cos(gyaw)
    ny = math.sin(gyaw)

    # Project previous and current positions onto gate normal
    prev_d = (self.prev_pos[0] - gpos[0]) * nx + (self.prev_pos[1] - gpos[1]) * ny
    curr_d = (pos[0] - gpos[0]) * nx + (pos[1] - gpos[1]) * ny

    # Crossed from negative side (approach) to positive side (through)
    if prev_d < 0 and curr_d >= 0:
      # Lateral offset in gate-local frame (perpendicular to normal)
      lateral = -(pos[0] - gpos[0]) * ny + (pos[1] - gpos[1]) * nx
      vertical = pos[2] - gpos[2]
      if abs(lateral) < self.gate_w / 2.0 and abs(vertical) < self.gate_h / 2.0:
        return True
    return False

  def _check_collision(self, pos):
    """Check collision with gate bars or ground."""
    if pos[2] < 0.1 and self.step_count >= TAKEOFF_GRACE:
      return True
    for bid in self.gate_bar_ids:
      contacts = p.getContactPoints(bodyA=self.drone_id, bodyB=bid)
      if len(contacts) > 0:
        return True
    return False

  def set_gate_size(self, w, h):
    """Update gate dimensions (for curriculum learning)."""
    self.gate_w = w
    self.gate_h = h

  def close(self):
    p.disconnect(self.physics_client)
