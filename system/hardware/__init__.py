from typing import cast

from openpilot.system.hardware.base import HardwareBase
from openpilot.system.hardware.pc.hardware import Pc

TICI = False
AGNOS = False
PC = True

HARDWARE = cast(HardwareBase, Pc())
