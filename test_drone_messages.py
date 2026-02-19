#!/usr/bin/env python3
"""
Test script to verify drone messaging structures are properly defined.
This script tests that the new DroneControl, DroneState, and GateData
messages can be created and used.

Run this after compiling cereal with scons.
"""

import sys


def test_drone_messages():
  try:
    import cereal.messaging as messaging
    from cereal import log

    print("✓ Successfully imported cereal modules")

    # Test DroneControl message
    print("\n--- Testing DroneControl ---")
    msg = messaging.new_message('droneControl')
    msg.droneControl.roll = 10.5
    msg.droneControl.pitch = -5.2
    msg.droneControl.yaw = 2.0
    msg.droneControl.throttle = 0.75
    msg.droneControl.armed = True
    msg.droneControl.maneuverType = log.DroneControl.ManeuverType.acro

    print(f"✓ DroneControl message created successfully")
    print(f"  Roll: {msg.droneControl.roll} deg/s")
    print(f"  Pitch: {msg.droneControl.pitch} deg/s")
    print(f"  Yaw: {msg.droneControl.yaw} deg/s")
    print(f"  Throttle: {msg.droneControl.throttle}")
    print(f"  Armed: {msg.droneControl.armed}")
    print(f"  Maneuver Type: {msg.droneControl.maneuverType}")

    # Test DroneState message
    print("\n--- Testing DroneState ---")
    msg2 = messaging.new_message('droneState')
    msg2.droneState.position = [1.0, 2.0, 3.0]
    msg2.droneState.velocity = [0.5, 0.0, 0.1]
    msg2.droneState.attitude = [1.0, 0.0, 0.0, 0.0]  # Quaternion [w, x, y, z]
    msg2.droneState.angularRates = [0.0, 0.0, 0.0]  # [P, Q, R]
    msg2.droneState.batteryVoltage = 14.8
    msg2.droneState.batteryPercent = 0.85
    msg2.droneState.armed = True
    msg2.droneState.flightMode = "ACRO"

    print(f"✓ DroneState message created successfully")
    print(f"  Position: {msg2.droneState.position}")
    print(f"  Velocity: {msg2.droneState.velocity}")
    print(f"  Attitude (quaternion): {msg2.droneState.attitude}")
    print(f"  Angular Rates: {msg2.droneState.angularRates}")
    print(f"  Battery Voltage: {msg2.droneState.batteryVoltage}V")
    print(f"  Battery Percent: {msg2.droneState.batteryPercent * 100}%")
    print(f"  Flight Mode: {msg2.droneState.flightMode}")

    # Test GateData in gateDetection message
    print("\n--- Testing GateData (in gateDetection) ---")
    msg3 = messaging.new_message('gateDetection', 2)  # 2 gates detected

    # First gate
    gate1 = msg3.gateDetection[0]
    gate1.gatePosition = [5.0, 0.0, 1.5]
    gate1.gateDimensions = [1.0, 1.0]
    gate1.confidence = 0.95
    gate1.gateYaw = 0.0
    gate1.gateId = 1
    gate1.distance = 5.0

    # Second gate
    gate2 = msg3.gateDetection[1]
    gate2.gatePosition = [10.0, 2.0, 1.5]
    gate2.gateDimensions = [1.2, 1.2]
    gate2.confidence = 0.87
    gate2.gateYaw = 0.2
    gate2.gateId = 2
    gate2.distance = 10.2

    print(f"✓ GateDetection message created with {len(msg3.gateDetection)} gates")
    for i, gate in enumerate(msg3.gateDetection):
      print(f"  Gate {i + 1}:")
      print(f"    Position: {gate.gatePosition}")
      print(f"    Dimensions: {gate.gateDimensions}")
      print(f"    Confidence: {gate.confidence}")
      print(f"    Distance: {gate.distance}m")

    # Test ModelDataV2 with gates
    print("\n--- Testing GateData in ModelDataV2 ---")
    msg4 = messaging.new_message('modelV2')
    msg4.modelV2.frameId = 12345

    # Initialize gates list with 1 gate
    msg4.modelV2.gates = [{}]  # Create one GateData entry
    msg4.modelV2.gates[0].gatePosition = [3.0, 0.5, 2.0]
    msg4.modelV2.gates[0].gateDimensions = [1.0, 1.0]
    msg4.modelV2.gates[0].confidence = 0.92
    msg4.modelV2.gates[0].gateId = 0
    msg4.modelV2.gates[0].distance = 3.16

    # Set next gate
    msg4.modelV2.nextGate.gatePosition = [3.0, 0.5, 2.0]
    msg4.modelV2.nextGate.gateDimensions = [1.0, 1.0]
    msg4.modelV2.nextGate.confidence = 0.92
    msg4.modelV2.nextGate.gateId = 0
    msg4.modelV2.nextGate.distance = 3.16

    print(f"✓ ModelDataV2 message with gates created successfully")
    print(f"  Frame ID: {msg4.modelV2.frameId}")
    print(f"  Number of gates detected: {len(msg4.modelV2.gates)}")
    print(f"  Next gate position: {msg4.modelV2.nextGate.gatePosition}")
    print(f"  Next gate distance: {msg4.modelV2.nextGate.distance}m")

    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("=" * 60)
    print("\nDrone messaging structures are correctly defined and working!")
    return 0

  except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease compile cereal first:")
    print("  cd /home/yash/openfighterpilot/cereal")
    print("  scons -j$(nproc)")
    return 1
  except AttributeError as e:
    print(f"✗ Attribute error: {e}")
    print("\nThe drone structures may not have been added correctly to cereal.")
    return 1
  except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
    return 1


if __name__ == "__main__":
  sys.exit(test_drone_messages())
