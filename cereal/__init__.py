import os
import capnp
from importlib.resources import as_file, files

capnp.remove_import_hook()

with as_file(files("cereal")) as fspath:
  CEREAL_PATH = fspath.as_posix()
  log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
  custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"))

  # car.capnp removed (broken symlink to deleted opendbc_repo)
  car_capnp = os.path.join(CEREAL_PATH, "car.capnp")
  if os.path.exists(car_capnp):
    car = capnp.load(car_capnp)
  else:
    car = None
