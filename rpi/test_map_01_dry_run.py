"""
Plans map_01.json with the real plan_mission(), then drives it through the
real MotorController.execute_leg()/execute_command() routing logic - the
exact same code main.py calls on a live run - except the serial port is
FakeSerial (from test_motor_controller.py), so no real STM32 is needed.
Prints (and writes to map_01_stm_commands.txt) the exact ASCII command
sequence the STM would actually receive.

Deliberately reuses MotorController's own routing rather than re-deriving
"which ASCII string would this Command produce" independently - a second,
hand-rolled implementation of that logic is exactly how this repo's old
testing/pathing.py and rpi/dry_run.py drifted from the real code path (see
their git history). If motor_controller.py's routing changes, this test's
output changes with it automatically instead of silently going stale.

Run with: pytest test_map_01_dry_run.py -s (from inside rpi/) - the -s is
required to see the printed command listing; without it pytest captures
stdout and only shows it on failure. Plain "pytest" (collecting the whole
rpi/ directory) works too. Running from outside rpi/ (e.g. the repo root)
raises ModuleNotFoundError: No module named 'motor_controller' - pytest's
import mechanism only adds this file's own directory (rpi/) to sys.path
when rpi/ is where collection actually starts.
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
# controller._ser.sent means execute_command() routed to something bogus,
# not a real protocol command.
KNOWN_PREFIXES = ("HELLO", "STRAIGHT,", "TURN,", "STEER,", "MOTOR,", "ENC,", "STOP")


def _load_scenario():
    with MAP_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return parse_scenario(data)


def test_map_01_produces_valid_stm_command_sequence(monkeypatch):
    monkeypatch.setattr(mc.serial, "Serial", FakeSerial)
    robot, obstacles = _load_scenario()

    try:
        mission = plan_mission(robot, obstacles)
    except PlanningError as exc:
        raise AssertionError(f"map_01.json has no reachable obstacles at all: {exc}")
    assert not mission.skipped_ids, f"map_01.json should be fully reachable, but skipped: {mission.skipped_ids}"

    controller = mc.MotorController()  # FakeSerial under the hood - handshake already sent
    for leg in mission.legs:
        controller.execute_leg(leg.commands)
    controller.close()

    # controller._ser.sent is already exactly what went out over the wire, in
    # order (handshake, every STRAIGHT/TURN/raw-fallback command and STATUS
    # poll, final STOP) - no leg/handshake/shutdown labels mixed in, since
    # those aren't things the STM ever actually receives.
    output = "\n".join(controller._ser.sent)
    print("\n" + output)
    OUTPUT_PATH.write_text(output + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT_PATH}")

    for line in controller._ser.sent:
        assert line.startswith(KNOWN_PREFIXES), f"unrecognised command sent to STM: {line!r}"
