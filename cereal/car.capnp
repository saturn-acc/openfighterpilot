using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

# Minimal stub for car types referenced by log.capnp.
# The real car.capnp lived in opendbc_repo which is removed for the drone fork.

@0x9de496db5dd0eb81;

struct OnroadEventDEPRECATED {
  name @0 :UInt16;
}

struct CarParams {
  safetyModel @0 :SafetyModel;

  enum SafetyModel {
    silent @0;
    allOutput @1;
  }
}

struct RadarData {
  errors @0 :List(Error);
  errorsDEPRECATED @1 :List(ErrorDEPRECATED);

  enum Error {
    canError @0;
    fault @1;
    wrongConfig @2;
  }

  enum ErrorDEPRECATED {
    canError @0;
  }
}

struct CarControl {
  actuators @0 :Actuators;
  hudControl @1 :HUDControl;

  struct Actuators {
    longControlState @0 :LongControlState;

    enum LongControlState {
      off @0;
      pid @1;
      stopping @2;
    }
  }

  struct HUDControl {
    audibleAlert @0 :AudibleAlert;
    visualAlert @1 :VisualAlert;

    enum AudibleAlert {
      none @0;
      engage @1;
      disengage @2;
      refuse @3;
      warningSoft @4;
      warningImmediate @5;
      prompt @6;
      promptRepeat @7;
      promptDistracted @8;
    }

    enum VisualAlert {
      none @0;
      fcw @1;
      steerRequired @2;
      brakePressed @3;
      wrongGear @4;
      seatbeltUnbuckled @5;
      speedTooHigh @6;
      ldw @7;
    }
  }
}

struct CarState {
  dummy @0 :UInt8;
}

struct CarOutput {
  dummy @0 :UInt8;
}
