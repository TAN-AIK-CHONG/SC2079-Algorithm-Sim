"""
Plans a scenario and prints the exact ASCII commands the STM would receive -
without needing a live serial connection. For handing a command sheet to
someone testing the STM by hand in PuTTY (no RPi in the loop).

Mirrors motor_controller.py's execute_command() routing decision exactly
(same imported constants - STRAIGHT_MIN_CM/MAX_CM, TURN_MIN_ANGLE_DEG,
TURNING_RADIUS_CM, SERVO_CENTER_US, RAW_STEER_LEFT_US/RIGHT_US,
RAW_DRIVE_PWM) so this can't silently drift from what the real code would
actually send.

Usage: python dry_run.py <scenario.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import parse_scenario  # noqa: E402
from planner import plan_mission, PlanningError  # noqa: E402
import motor_controller as mc  # noqa: E402 - reused for its constants only, no serial connection opened


def command_to_lines(command) -> list[str]:
    """Returns the PuTTY-typeable line(s) for one Command, in order, plus a
    plain-English note on what to watch for before moving to the next one."""
    direction = command.direction
    turn = command.turn
    lines = []

    if direction == "FORWARD" and turn == "STRAIGHT":
        cm = round(command.distance_cm)
        if mc.STRAIGHT_MIN_CM <= cm <= mc.STRAIGHT_MAX_CM:
            lines.append(f"STRAIGHT,GOCM,{cm}")
            lines.append("STRAIGHT,STATUS   <- repeat until state=DONE, then continue")
            return lines
    elif direction == "FORWARD":
        degrees = round(command.swept_angle_deg)
        if degrees >= mc.TURN_MIN_ANGLE_DEG:
            radius_mm = round(mc.TURNING_RADIUS_CM * 10)
            signed_degrees = degrees if turn == "RIGHT" else -degrees
            lines.append(f"TURN,START,R={radius_mm},A={signed_degrees}")
            lines.append("TURN,STATUS      <- repeat until state=DONE, then continue")
            return lines

    # Raw fallback: backward commands, or forward commands too short/shallow
    # for STRAIGHT,GOCM/TURN,START. No single self-stopping command exists
    # for this on the STM yet - drive, watch ENC,GET,CM by hand, STOP at target.
    steer_us = {
        "STRAIGHT": mc.SERVO_CENTER_US,
        "LEFT": mc.RAW_STEER_LEFT_US,
        "RIGHT": mc.RAW_STEER_RIGHT_US,
    }[turn]
    pwm = mc.RAW_DRIVE_PWM if direction == "FORWARD" else -mc.RAW_DRIVE_PWM
    target_cm = command.distance_cm
    lines.append(f"STEER,US,{steer_us}")
    lines.append("ENC,RESET")
    lines.append(f"MOTOR,B,{pwm},{pwm}")
    lines.append(
        f"ENC,GET,CM       <- repeat FAST (under 1s between polls - see heartbeat"
        f" warning below); STOP once (l_cm+r_cm)/2 reaches {target_cm}cm "
        f"(going {'backward, values will read negative' if direction == 'REVERSE' else 'forward'})"
    )
    lines.append("STOP")
    return lines


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: python {Path(__file__).name} <scenario.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)
    robot, obstacles = parse_scenario(data)

    print("=" * 60)
    print("SCENARIO")
    print("=" * 60)
    print(f"robot start : grid ({data['robot']['x_coord']}, {data['robot']['y_coord']}) facing {data['robot']['facing']}")
    for obs in data["obstacles"]:
        print(f"obstacle {obs['id']:<3}: grid ({obs['x_coord']}, {obs['y_coord']}), image faces {obs['image_side']}")
    print()

    try:
        plan = plan_mission(robot, obstacles)
    except PlanningError as exc:
        print(f"planning failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if plan.skipped_ids:
        print(f"NOTE: obstacle(s) {plan.skipped_ids} could not be reached and are skipped.\n")

    uses_raw_fallback = any(
        command.direction != "FORWARD"
        or (command.turn == "STRAIGHT" and not (mc.STRAIGHT_MIN_CM <= round(command.distance_cm) <= mc.STRAIGHT_MAX_CM))
        or (command.turn != "STRAIGHT" and round(command.swept_angle_deg) < mc.TURN_MIN_ANGLE_DEG)
        for leg in plan.legs
        for command in leg.commands
    )

    print("=" * 60)
    print("COMMANDS - type each line into PuTTY in order, top to bottom")
    print("=" * 60)
    print("First: HELLO, then HB, then confirm robot is on the ground with clear space.")
    print()
    print("*** IMPORTANT: the FIRST STRAIGHT,GOCM or TURN,START command you send ***")
    print("*** auto-runs a ~5s gyro calibration first. The robot will NOT move    ***")
    print("*** right away - keep it completely still and don't touch it for      ***")
    print("*** ~5 seconds after that first command. Every command after the      ***")
    print("*** first reuses that calibration and does not re-trigger it.         ***")
    if uses_raw_fallback:
        print()
        print("*** IMPORTANT: this plan includes raw MOTOR,B step(s) (see 'STEER,US' ***")
        print("*** below). Those have NO self-stopping command yet, and the STM's    ***")
        print("*** 1-second heartbeat watchdog will auto-STOP them (and print an     ***")
        print("*** unsolicited ERR,HB_TIMEOUT) if more than ~1s passes between your  ***")
        print("*** commands - poll ENC,GET,CM quickly and repeatedly. If it auto-    ***")
        print("*** stops early, that's the watchdog working as intended, not a bug - ***")
        print("*** just check the encoder reading and re-send MOTOR,B to continue.   ***")
    print()

    step = 1
    for leg in plan.legs:
        print(f"--- leg: {leg.from_id} -> obstacle {leg.to_id} ---")
        for command in leg.commands:
            angle_note = f", {command.swept_angle_deg} deg" if command.swept_angle_deg is not None else ""
            print(f"[{step}] {command.direction} {command.turn} {command.distance_cm}cm{angle_note}")
            for line in command_to_lines(command):
                print(f"    > {line}")
            step += 1
        print(f"    ** at this point the robot should be at obstacle {leg.to_id}'s viewing pose - take the photo now **")
        print()

    print("STOP   <- send at the very end regardless")


if __name__ == "__main__":
    main()
