"""
Plans map_01.json with the real plan_mission(), then drives it through the
real MotorController.execute_leg()/execute_command() routing logic - the
exact same code main.py calls on a live run - except the serial port is
FakeSerial (from test_motor_controller.py), so no real STM32 is needed.
Prints, and writes to map_01_stm_commands.txt, the exact ASCII command
sequence the STM would actually receive.

Deliberately reuses MotorController's own routing rather than re-deriving
"which ASCII string would this Command produce" independently - a second,
hand-rolled implementation of that logic is exactly how this repo's old
testing/pathing.py and rpi/dry_run.py drifted from the real code path (see
their git history). If motor_controller.py's routing changes, this output
changes with it automatically instead of silently going stale.

Two ways to run, both from inside rpi/:
  python test_map_01_dry_run.py        # just prints + writes the .txt
  pytest test_map_01_dry_run.py -s     # same, plus asserts every command is
                                       # a real protocol command (-s to see
                                       # the listing; pytest hides stdout
                                       # otherwise)
Running from outside rpi/ raises ModuleNotFoundError: No module named
'motor_controller' - only rpi/ is on sys.path when rpi/ is the working dir.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import motor_controller as mc
from model import parse_scenario
from planner import PlanningError, plan_mission
from test_motor_controller import FakeSerial

MAP_PATH = Path(__file__).resolve().parent / "map_01.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "map_01_stm_commands.txt"

# Recognised command names from PROTOCOL_en.md - anything else appearing in
# the sent list means execute_command() routed to something bogus, not a
# real protocol command.
KNOWN_PREFIXES = ("HELLO", "STRAIGHT,", "TURN,", "STEER,", "MOTOR,", "ENC,", "STOP")


def stm_command_sequence() -> list[str]:
    """The exact ordered list of ASCII lines MotorController would send the
    STM for map_01.json - handshake, every STRAIGHT/TURN/raw-fallback
    command and its STATUS polls, the final STOP. No labels or blank lines:
    only what actually goes over the wire."""
    with MAP_PATH.open(encoding="utf-8") as f:
        robot, obstacles = parse_scenario(json.load(f))

    mission = plan_mission(robot, obstacles)  # PlanningError propagates - map_01 is solvable
    if mission.skipped_ids:
        raise RuntimeError(f"map_01.json is not fully reachable, skipped: {mission.skipped_ids}")

    mc.serial.Serial = FakeSerial  # no real STM32 - FakeSerial records every write
    controller = mc.MotorController()
    for leg in mission.legs:
        controller.execute_leg(leg.commands)
    controller.close()
    return list(controller._ser.sent)


def _write_and_print(commands: list[str]) -> None:
    text = "\n".join(commands)
    print(text)
    OUTPUT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT_PATH}")


def test_map_01_produces_valid_stm_command_sequence():
    try:
        commands = stm_command_sequence()
    except (PlanningError, RuntimeError) as exc:
        raise AssertionError(str(exc))

    _write_and_print(commands)
    for line in commands:
        assert line.startswith(KNOWN_PREFIXES), f"unrecognised command sent to STM: {line!r}"


if __name__ == "__main__":
    _write_and_print(stm_command_sequence())
