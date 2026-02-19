import math
from enum import StrEnum, auto

import cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.locationd.helpers import Pose

# Constants previously from opendbc.car (car modules removed)
ACCELERATION_DUE_TO_GRAVITY = 9.81  # m/s^2
ISO_LATERAL_ACCEL = 3.0  # m/s^2
ACCEL_MIN = -3.5  # m/s^2
ACCEL_MAX = 2.0   # m/s^2

MIN_EXCESSIVE_ACTUATION_COUNT = int(0.25 / DT_CTRL)
MIN_LATERAL_ENGAGE_BUFFER = int(1 / DT_CTRL)


class ExcessiveActuationType(StrEnum):
  LONGITUDINAL = auto()
  LATERAL = auto()


class ExcessiveActuationCheck:
  def __init__(self):
    self._excessive_counter = 0
    self._engaged_counter = 0

  def update(self, sm: messaging.SubMaster, CS, calibrated_pose: Pose) -> ExcessiveActuationType | None:
    # Stubbed for drone mode — no car-specific actuation checks needed
    return None
