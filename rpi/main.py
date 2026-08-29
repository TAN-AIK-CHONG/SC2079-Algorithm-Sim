"""
Runs ON THE RPI. This is the whole pipeline in one process now - no HTTP
call to a separate algorithm server: plan the mission by calling into
src/planner.py directly, then drive each leg through the STM32.

Usage: python main.py <scenario.json>

`scenario.json` stands in for however the robot/obstacles actually get
built on the real run (e.g. obstacle positions + image_side from Aasim's
image recognition, robot's own start position). Swap load_scenario() out
for that once it's ready - everything below it stays the same.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import Obstacle, Robot, parse_scenario  # noqa: E402
from planner import plan_mission, PlanningError  # noqa: E402
from motor_controller import MotorController  # noqa: E402


def load_scenario(path: Path) -> tuple[Robot, list[Obstacle]]:
    with open(path) as f:
        data = json.load(f)
    return parse_scenario(data)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: python {Path(__file__).name} <scenario.json>", file=sys.stderr)
        sys.exit(1)

    robot, obstacles = load_scenario(Path(sys.argv[1]))

    # plan_mission() returns list[Leg] (see src/planner.py) - each Leg has
    # .from_id/.to_id/.commands. It can also raise PlanningError if
    # hybrid_astar found no collision-free path for some leg (e.g. obstacles
    # placed too close together, or too close to a wall for this radius).
    try:
        legs = plan_mission(robot, obstacles)
    except PlanningError as exc:
        print(f"planning failed: {exc}", file=sys.stderr)
        sys.exit(1)

    total_cm = sum(command.distance_cm for leg in legs for command in leg.commands)
    order = [legs[0].from_id, *(leg.to_id for leg in legs)] if legs else []
    print(f"visiting order: {order}  ({total_cm}cm total)")

    with MotorController() as controller:
        for leg in legs:
            print(f"-> {leg.to_id} ({len(leg.commands)} commands)")
            controller.execute_leg(leg.commands)

    print("mission complete")


if __name__ == "__main__":
    main()
