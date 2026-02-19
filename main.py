#!/usr/bin/env python3
"""
OpenFighterPilot — main entry point.

Launches the PyBullet drone simulation, flight controller, and cereal
messaging as coordinated processes.

Usage:
  python main.py                        # headless, gravity compensator
  python main.py --gui                  # PyBullet GUI window
  python main.py --controller dummy     # self-contained dummy planner demo
  python main.py --gui --seconds 30     # GUI + time limit
"""
import argparse
import importlib
import os
import signal
import sys
import time
from multiprocessing import Process



# ---------------------------------------------------------------------------
# Process wrappers
# ---------------------------------------------------------------------------

def _run_module(module_path, name, extra_argv=None):
  """Import *module_path*, reset cereal context, then call module.main()."""
  import cereal.messaging as messaging
  messaging.reset_context()

  # Allow child modules that use argparse to see the flags we pass in
  if extra_argv is not None:
    sys.argv = [name] + extra_argv

  mod = importlib.import_module(module_path)
  mod.main()


def launch(module_path, name, extra_argv=None):
  """Spawn *module_path* as a child process and return the Process handle."""
  p = Process(
    target=_run_module,
    args=(module_path, name, extra_argv),
    name=name,
    daemon=True,
  )
  p.start()
  return p


# ---------------------------------------------------------------------------
# Monitoring helpers
# ---------------------------------------------------------------------------

def wait_for_message(service, timeout=5.0):
  """Block until *service* appears on the bus or *timeout* expires."""
  import cereal.messaging as messaging
  sm = messaging.SubMaster([service])
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    sm.update(100)  # poll every 100 ms
    if sm.updated[service]:
      return True
  return False


def print_status(sm):
  """Print a one-liner with drone telemetry from a SubMaster snapshot."""
  if not sm.updated.get('droneState'):
    return
  ds = sm['droneState']
  pos = list(ds.position)
  vel = list(ds.velocity)
  bat = ds.batteryPercent
  armed = ds.armed
  mode = ds.flightMode
  print(
    f"  pos=[{pos[0]:+7.2f} {pos[1]:+7.2f} {pos[2]:+7.2f}]  "
    f"vel=[{vel[0]:+6.2f} {vel[1]:+6.2f} {vel[2]:+6.2f}]  "
    f"bat={bat:.0%}  armed={armed}  mode={mode}",
    end="\r",
  )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BANNER = r"""
   ___                   ___ _      _   _            ___ _ _     _
  / _ \ _ __  ___ _ _   | __(_)__ _| |_| |_ ___ _ _ | _ (_) |___| |_
 | (_) | '_ \/ -_) ' \  | _|| / _` | ' \  _/ -_) '_||  _/ | / _ \  _|
  \___/| .__/\___|_||_| |_| |_\__, |_||_|\__\___|_| |_| |_|_\___/\__|
       |_|                    |___/
"""


def main():
  parser = argparse.ArgumentParser(
    description="OpenFighterPilot — drone racing on cereal + PyBullet",
  )
  parser.add_argument("--gui", action="store_true",
                      help="Open PyBullet GUI window")
  parser.add_argument("--controller", choices=["gravity", "dummy", "race", "rl"],
                      default="gravity",
                      help="Which flight controller to run (default: gravity)")
  parser.add_argument("--seconds", type=float, default=0,
                      help="Stop after N seconds (0 = run forever)")
  parser.add_argument("--mass", type=float, default=1.0,
                      help="Quadrotor mass in kg")
  parser.add_argument("--max-thrust", type=float, default=20.0,
                      help="Maximum thrust in N")
  args = parser.parse_args()

  print(BANNER)
  print(f"  Controller : {args.controller}")
  print(f"  GUI        : {'on' if args.gui else 'off'}")
  print(f"  Mass       : {args.mass} kg")
  print(f"  Max thrust : {args.max_thrust} N")
  if args.seconds:
    print(f"  Duration   : {args.seconds} s")
  else:
    print(f"  Duration   : unlimited (Ctrl+C to stop)")
  print()

  procs: list[Process] = []

  # ------------------------------------------------------------------
  # Mode 1: dummy planner — self-contained single-process demo
  # ------------------------------------------------------------------
  if args.controller == "dummy":
    bridge_argv = []
    if args.gui:
      bridge_argv.append("--gui")
    bridge_argv.extend(["--seconds", str(args.seconds or 10)])

    print("[main] Starting dummy planner (single-process demo)...")
    proc = launch("selfdrive.controls.dummy_planner", "dummy_planner", bridge_argv)
    procs.append(proc)

  # ------------------------------------------------------------------
  # Mode 2: bridge + gravity compensator (multi-process architecture)
  # ------------------------------------------------------------------
  elif args.controller == "gravity":
    # 1. Start simulation bridge
    bridge_argv = []
    if args.gui:
      bridge_argv.append("--gui")
    bridge_argv.extend(["--mass", str(args.mass)])
    bridge_argv.extend(["--max-thrust", str(args.max_thrust)])

    print("[main] Starting drone bridge (PyBullet sim)...")
    bridge = launch("tools.sim.bridge_drone", "drone_bridge", bridge_argv)
    procs.append(bridge)

    # Wait for the bridge to start publishing droneState
    print("[main] Waiting for droneState on the bus...", end=" ", flush=True)
    if wait_for_message("droneState"):
      print("OK")
    else:
      print("TIMEOUT (continuing anyway)")

    # 2. Start gravity compensator
    print("[main] Starting gravity compensator...")
    grav = launch("selfdrive.controls.gravity_compensator", "gravity_compensator")
    procs.append(grav)

    # Wait for control messages
    print("[main] Waiting for droneControl on the bus...", end=" ", flush=True)
    if wait_for_message("droneControl"):
      print("OK")
    else:
      print("TIMEOUT (continuing anyway)")

  # ------------------------------------------------------------------
  # Mode 3: race — full 4-process racing stack
  # ------------------------------------------------------------------
  elif args.controller == "race":
    # 1. Start enhanced simulation bridge with gates
    bridge_argv = []
    if args.gui:
      bridge_argv.append("--gui")
    bridge_argv.extend(["--mass", str(args.mass)])
    bridge_argv.extend(["--max-thrust", str(args.max_thrust)])

    print("[main] Starting bridge_pybullet (PyBullet sim + gates)...")
    bridge = launch("tools.sim.bridge_pybullet", "bridge_pybullet", bridge_argv)
    procs.append(bridge)

    print("[main] Waiting for droneState on the bus...", end=" ", flush=True)
    if wait_for_message("droneState"):
      print("OK")
    else:
      print("TIMEOUT (continuing anyway)")

    # 2. Start planning process
    print("[main] Starting plannerd...")
    plannerd = launch("selfdrive.controls.plannerd", "plannerd")
    procs.append(plannerd)

    print("[main] Waiting for dronePlan on the bus...", end=" ", flush=True)
    if wait_for_message("dronePlan"):
      print("OK")
    else:
      print("TIMEOUT (continuing anyway)")

    # 3. Start control process
    print("[main] Starting controld...")
    controld = launch("selfdrive.controls.controld", "controld")
    procs.append(controld)

    print("[main] Waiting for droneControl on the bus...", end=" ", flush=True)
    if wait_for_message("droneControl"):
      print("OK")
    else:
      print("TIMEOUT (continuing anyway)")

  # ------------------------------------------------------------------
  # Mode 4: rl — bridge + RL policy (no controld)
  # ------------------------------------------------------------------
  elif args.controller == "rl":
    # 1. Start enhanced simulation bridge with gates
    bridge_argv = []
    if args.gui:
      bridge_argv.append("--gui")
    bridge_argv.extend(["--mass", str(args.mass)])
    bridge_argv.extend(["--max-thrust", str(args.max_thrust)])

    print("[main] Starting bridge_pybullet (PyBullet sim + gates)...")
    bridge = launch("tools.sim.bridge_pybullet", "bridge_pybullet", bridge_argv)
    procs.append(bridge)

    print("[main] Waiting for droneState on the bus...", end=" ", flush=True)
    if wait_for_message("droneState"):
      print("OK")
    else:
      print("TIMEOUT (continuing anyway)")

    # 2. Start plannerd in RL mode (publishes droneControl directly, no controld)
    print("[main] Starting plannerd --rl...")
    plannerd = launch("selfdrive.controls.plannerd", "plannerd", ["--rl"])
    procs.append(plannerd)

    print("[main] Waiting for droneControl on the bus...", end=" ", flush=True)
    if wait_for_message("droneControl"):
      print("OK")
    else:
      print("TIMEOUT (continuing anyway)")

  # ------------------------------------------------------------------
  # Monitor loop
  # ------------------------------------------------------------------
  print()
  print("[main] All processes running. Press Ctrl+C to stop.")
  print("-" * 72)

  import cereal.messaging as messaging
  sm = messaging.SubMaster(['droneState'])
  deadline = (time.monotonic() + args.seconds) if args.seconds else None

  try:
    while True:
      # Check children are alive
      for proc in procs:
        if not proc.is_alive():
          print(f"\n[main] Process '{proc.name}' exited (code={proc.exitcode})")
          raise SystemExit(proc.exitcode or 0)

      # Print telemetry
      sm.update(200)
      print_status(sm)

      # Time limit
      if deadline and time.monotonic() >= deadline:
        print(f"\n[main] {args.seconds}s elapsed — shutting down.")
        break

  except KeyboardInterrupt:
    print("\n[main] Ctrl+C received — shutting down.")

  # ------------------------------------------------------------------
  # Cleanup
  # ------------------------------------------------------------------
  for proc in procs:
    if proc.is_alive():
      os.kill(proc.pid, signal.SIGINT)

  # Give children a moment to clean up, then force-kill stragglers
  for proc in procs:
    proc.join(timeout=3.0)
    if proc.is_alive():
      print(f"[main] Force-killing '{proc.name}'")
      proc.kill()

  print("[main] Done.")


if __name__ == "__main__":
  main()
