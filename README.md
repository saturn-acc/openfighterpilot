<div align="center">

# OpenFighterPilot

**Autonomous drone racing framework built on cereal messaging and PyBullet simulation.**

A multi-process flight control stack for developing and testing drone racing algorithms,
forked from [openpilot](https://github.com/commaai/openpilot).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## Overview

OpenFighterPilot is a real-time drone racing platform with three layers of complexity:

- **Gravity mode** — 2-process hover controller for basic testing
- **Dummy mode** — single-process self-contained demo
- **Race mode** — full 3-process pipeline with FPV camera, planning, and PID tracking control

All inter-process communication uses [cereal](https://github.com/commaai/cereal) (Cap'n Proto over shared memory), giving each module a clean pub/sub interface at fixed rates.

## Architecture

```
Race Mode (3-process):

  bridge_pybullet ──droneState (100Hz)──────> plannerd, controld
  bridge_pybullet ──droneCameraState (20Hz)──> [future vision pipeline]
  bridge_pybullet ──gateDetection (20Hz)────> [future vision pipeline]
  plannerd ────────dronePlan (20Hz)─────────> controld
  controld ────────droneControl (100Hz)─────> bridge_pybullet
```

```
Gravity Mode (2-process):

  bridge_drone ────droneState (100Hz)───> gravity_compensator
  gravity_compensator ─droneControl (100Hz)─> bridge_drone
```

## Quick Start

```bash
# Clone
git clone https://github.com/saturn-acc/openfighterpilot.git
cd openfighterpilot

# Install dependencies
pip install -e '.[drone]'

# Run headless (gravity hover)
python main.py --controller gravity --seconds 10

# Run the full racing stack
python main.py --controller race --seconds 20

# With PyBullet GUI (FPV chase cam)
python main.py --controller race --gui --seconds 20
```

## Controller Modes

| Mode | Processes | Description |
|------|-----------|-------------|
| `gravity` | bridge_drone + gravity_compensator | Hover-only. Compensates for tilt to maintain altitude. |
| `dummy` | single process | Self-contained demo. Validates cereal message round-trip. |
| `race` | bridge_pybullet + plannerd + controld | Full stack. Fly-forward planner + PID velocity tracking + FPV camera. |

### CLI Options

```
python main.py [OPTIONS]

--controller {gravity,dummy,race}   Flight mode (default: gravity)
--gui                               Open PyBullet GUI window
--seconds N                         Stop after N seconds (0 = forever)
--mass KG                           Quadrotor mass (default: 1.0)
--max-thrust N                      Maximum thrust in Newtons (default: 20.0)
```

## Key Modules

| Module | Role |
|--------|------|
| `tools/sim/bridge_pybullet.py` | PyBullet sim with gates, FPV camera (320x240 RGB at 20Hz), chase cam GUI |
| `tools/sim/bridge_drone.py` | Minimal PyBullet sim (no gates, no camera) |
| `selfdrive/controls/plannerd.py` | Fly-forward planner: 2 m/s forward + altitude hold at 1m |
| `selfdrive/controls/controld.py` | PID velocity controller (4 axes) at 100Hz |
| `selfdrive/controls/gravity_compensator.py` | Tilt-compensated hover: `thrust = mg / cos(tilt)` |
| `selfdrive/controls/visiond.py` | Gate detection to body-relative error vectors (reserved for future use) |
| `selfdrive/controls/dummy_planner.py` | Single-process demo (no messaging) |
| `cereal/log.capnp` | Cap'n Proto schema for all drone messages |
| `cereal/services.py` | Service registry (frequencies, queue sizes) |

## Message Types

| Message | Rate | Contents |
|---------|------|----------|
| `droneState` | 100Hz | Position, velocity, attitude (quaternion), angular rates, battery |
| `droneControl` | 100Hz | Throttle (0-1), roll/pitch/yaw commands, armed state |
| `dronePlan` | 20Hz | Desired velocity (world frame), desired yaw, target position |
| `droneCameraState` | 20Hz | Raw RGB image (320x240), frame ID, timestamp |
| `gateDetection` | 20Hz | Relative gate position, dimensions, confidence, distance |

## Project Structure

```
openfighterpilot/
  main.py                          # Entry point — launches processes by mode
  cereal/
    log.capnp                      # Message schemas (DroneState, DronePlan, etc.)
    services.py                    # Service frequencies and queue sizes
  selfdrive/controls/
    plannerd.py                    # Fly-forward planner (20Hz)
    controld.py                    # PID flight controller (100Hz)
    gravity_compensator.py         # Hover controller (100Hz)
    visiond.py                     # Vision pipeline (future use)
    dummy_planner.py               # Single-process demo
  tools/sim/
    bridge_pybullet.py             # Full sim: physics + gates + FPV camera
    bridge_drone.py                # Minimal sim: physics only
    assets/quadrotor.urdf          # Drone model (1kg, 24cm diagonal)
```

## Roadmap

- [x] Multi-process cereal messaging architecture
- [x] PyBullet physics simulation with URDF quadrotor
- [x] PID velocity tracking controller
- [x] FPV camera publishing (320x240 RGB at 20Hz)
- [x] Visual gate markers in simulation
- [ ] YOLOv8 gate detection from FPV images
- [ ] Vision-based planning (replace dummy fly-forward)
- [ ] Multi-gate race course
- [ ] Lap timing and scoring

## License

MIT &mdash; see [LICENSE](LICENSE) for details.

This project is forked from [openpilot](https://github.com/commaai/openpilot) by [comma.ai](https://comma.ai).
