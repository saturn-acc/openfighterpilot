import os
import operator
import platform

from openpilot.common.params import Params
from openpilot.system.hardware import PC, TICI
from openpilot.system.manager.process import PythonProcess, NativeProcess, DaemonProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None


def driverview(started: bool, params: Params) -> bool:
  return started or params.get_bool("IsDriverViewEnabled")


def logging(started: bool, params: Params) -> bool:
  run = not params.get_bool("DisableLogging")
  return started and run


def ublox_available() -> bool:
  return os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')


def ublox(started: bool, params: Params) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox)
  return started and use_ublox


def qcomgps(started: bool, params: Params) -> bool:
  return started and not ublox_available()


def always_run(started: bool, params: Params) -> bool:
  return True


def only_onroad(started: bool, params: Params) -> bool:
  return started


def only_offroad(started: bool, params: Params) -> bool:
  return not started


def or_(*fns):
  return lambda *args: operator.or_(*(fn(*args) for fn in fns))


def and_(*fns):
  return lambda *args: operator.and_(*(fn(*args) for fn in fns))


procs = [
  DaemonProcess("manage_athenad", "system.athena.manage_athenad", "AthenadPid"),

  NativeProcess("loggerd", "system/loggerd", ["./loggerd"], logging),
  NativeProcess("encoderd", "system/loggerd", ["./encoderd"], only_onroad),
  PythonProcess("logmessaged", "system.logmessaged", always_run),

  NativeProcess("camerad", "system/camerad", ["./camerad"], driverview, enabled=not WEBCAM),
  PythonProcess("webcamerad", "tools.webcam.camerad", driverview, enabled=WEBCAM),
  PythonProcess("proclogd", "system.proclogd", only_onroad, enabled=platform.system() != "Darwin"),
  PythonProcess("journald", "system.journald", only_onroad, platform.system() != "Darwin"),
  PythonProcess("timed", "system.timed", always_run, enabled=not PC),

  PythonProcess("modeld", "selfdrive.modeld.modeld", only_onroad),
  PythonProcess("dmonitoringmodeld", "selfdrive.modeld.dmonitoringmodeld", driverview, enabled=(WEBCAM or not PC)),

  PythonProcess("sensord", "system.sensord.sensord", only_onroad, enabled=not PC),
  PythonProcess("ui", "selfdrive.ui.ui", always_run, restart_if_crash=True),
  PythonProcess("soundd", "selfdrive.ui.soundd", driverview),
  PythonProcess("locationd", "selfdrive.locationd.locationd", only_onroad),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd", only_onroad),
  PythonProcess("selfdrived", "selfdrive.selfdrived.selfdrived", only_onroad),
  PythonProcess("deleter", "system.loggerd.deleter", always_run),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", driverview, enabled=(WEBCAM or not PC)),
  PythonProcess("qcomgpsd", "system.qcomgpsd.qcomgpsd", qcomgps, enabled=TICI),
  PythonProcess("ubloxd", "system.ubloxd.ubloxd", ublox, enabled=TICI),
  PythonProcess("pigeond", "system.ubloxd.pigeond", ublox, enabled=TICI),
  PythonProcess("hardwared", "system.hardware.hardwared", always_run),
  PythonProcess("tombstoned", "system.tombstoned", always_run, enabled=not PC),
  PythonProcess("updated", "system.updated.updated", only_offroad, enabled=not PC),
  PythonProcess("uploader", "system.loggerd.uploader", always_run),
  PythonProcess("statsd", "system.statsd", always_run),
  PythonProcess("feedbackd", "selfdrive.ui.feedback.feedbackd", only_onroad),

  # drone processes
  PythonProcess("drone_bridge", "tools.sim.bridge_drone", only_onroad),
  PythonProcess("bridge_pybullet", "tools.sim.bridge_pybullet", only_onroad),
  PythonProcess("dummy_planner", "selfdrive.controls.dummy_planner", only_onroad),
  PythonProcess("plannerd", "selfdrive.controls.plannerd", only_onroad),
  PythonProcess("controld", "selfdrive.controls.controld", only_onroad),

  # debug procs
  NativeProcess("bridge", "cereal/messaging", ["./bridge"], only_onroad),
  PythonProcess("webrtcd", "system.webrtc.webrtcd", only_onroad),
]

managed_processes = {p.name: p for p in procs}
