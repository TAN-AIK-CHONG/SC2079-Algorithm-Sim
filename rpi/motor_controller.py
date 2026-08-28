"""
Runs ON THE RPI - translates the drive commands produced by src/planner.py
into the STM32 serial protocol. Called directly by main.py; there is no
network/API step, everything runs in one process.

Two execution paths, chosen per command in execute_command():

  - FORWARD commands use the STM's own closed-loop routines: STRAIGHT,GOCM
    for straight runs, TURN,START for turns. The STM handles PWM/servo/
    gyro-correction internally - this file only starts the routine and
    polls ...,STATUS until it reports DONE. No PWM/steering calibration
    needed on our side for these; the STM team has already tuned them
    (see drive_control.c: SERVO_CENTER_US=1712, BASE_PWM=2900 confirmed by
    field testing in test_distance.c).

  - Everything else falls back to the older raw MOTOR/STEER/ENC-polling
    approach:
      * BACKWARD (reverse) commands - neither STRAIGHT,GOCM nor TURN,START
        takes a direction argument (see drive_control.h), so reversing is
        only possible via raw MOTOR,B with negative PWM.
      * Forward commands outside STRAIGHT,GOCM/TURN,START's supported
        range - STRAIGHT,GOCM only accepts 10-500cm, TURN,START only
        accepts |angle| >= 5deg (both enforced by the STM, see
        drive_control.c's TURN_MIN_A_DEG / the STRAIGHT,GOCM bounds check
        in test_uart.c) - short segments would otherwise be rejected
        outright with ERR,BAD_VALUE.
    This path is NOT well calibrated: no file among the STM team's
    calibration builds tests reversing at all, so RAW_DRIVE_PWM/
    RAW_STEER_LEFT_US/RAW_STEER_RIGHT_US below are still guesses (the
    steering values are the STM's own "starting point, not final
    calibration" numbers from test_turn_calib.c).

KNOWN FIRMWARE BUG - verify before trusting TURN,START for both directions:
  As of mdp20260826afternoon_STRAIGHT_and_TURN.zip's drive_control.c,
  Turn_Start()'s steering pulse is
      delta_us = VEH_SERVO_TURN_SIGN * delta_deg / VEH_SERVO_DEG_PER_US
  which never references the SIGN of the requested angle_deg, and
  Drive_Step()'s turn phase compares |yaw| to |angle_deg| (magnitude
  only). So today TURN,START steers the same physical direction no
  matter what sign A is - it cannot yet actually turn both left and
  right. Flag this to whoever owns drive_control.c; the fix is to also
  multiply delta_us by the sign of the requested angle_deg. Until it's
  fixed, only one turn direction from this function will work correctly
  on the real robot.
"""

from __future__ import annotations

import os
import time

import serial
import serial.tools.list_ports


def _autodetect_port() -> str | None:
    """If exactly one serial device is plugged in, use it - avoids having to
    re-set MOTOR_SERIAL_PORT every time Windows hands the CH9102F a new COM
    number on replug. Ambiguous (0 or 2+ devices) -> None, caller falls back."""
    ports = list(serial.tools.list_ports.comports())
    return ports[0].device if len(ports) == 1 else None


# Priority: explicit MOTOR_SERIAL_PORT env var (set this if autodetect picks
# the wrong device, e.g. two serial adapters plugged in at once) -> the one
# connected serial device, if there's exactly one -> "COM5" as a last-resort
# fallback for whoever has neither.
SERIAL_PORT = os.environ.get("MOTOR_SERIAL_PORT") or _autodetect_port() or "COM5"
BAUD_RATE = 115200

# Confirmed from real hardware calibration (drive_control.c / test_uart.c,
# mdp20260826afternoon_STRAIGHT_and_TURN.zip) - shared by every path below.
SERVO_CENTER_US = 1712

# Must match algorithms/dubins.py's TURNING_RADIUS_CM - every turn command
# from planner.py assumes paths were planned at this radius.
TURNING_RADIUS_CM = 25

# STRAIGHT,GOCM / TURN,START limits enforced by the STM (drive_control.c).
# Commands outside these ranges are rejected outright, so we route them to
# the raw fallback instead of sending them and getting ERR,BAD_VALUE back.
STRAIGHT_MIN_CM = 10
STRAIGHT_MAX_CM = 500
TURN_MIN_ANGLE_DEG = 5

# --- Raw fallback path only (backward commands, or forward commands outside
# the ranges above). NOT confirmed - no calibration build tests reversing.
RAW_DRIVE_PWM = 2900              # reuses the confirmed forward BASE_PWM; unverified in reverse
RAW_STEER_LEFT_US = 1512           # test_turn_calib.c's own "starting point, not final" value
RAW_STEER_RIGHT_US = 1912          # ditto
# ---------------------------------------------------------------------------

DISTANCE_TOLERANCE_CM = 0.5    # raw fallback: stop once within this much of the target
POLL_INTERVAL_S = 0.02
COMMAND_TIMEOUT_S = 20.0        # safety net: abort a single command if it never completes


class MotorControllerError(RuntimeError):
    pass


class MotorController:
    """One instance per serial connection to the STM32. Use execute_leg() /
    execute_command() to drive the commands planner.plan_mission() returns."""

    def __init__(self, port: str = SERIAL_PORT, baudrate: int = BAUD_RATE, timeout: float = 1.0):
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        self._handshake()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._ser.close()

    def __enter__(self) -> "MotorController":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------- serial --

    def _send(self, command: str) -> str:
        self._ser.write((command + "\r\n").encode("ascii"))
        response = self._ser.readline().decode("ascii", errors="replace").strip()
        if not response:
            raise MotorControllerError(f"no response from STM for '{command}' (timed out)")
        if response.startswith("ERR"):
            raise MotorControllerError(f"STM rejected '{command}': {response}")
        return response

    def _handshake(self) -> None:
        response = self._send("HELLO")
        if not response.startswith("HELLO"):
            raise MotorControllerError(f"unexpected HELLO response: {response!r}")

    @staticmethod
    def _parse_fields(response: str) -> dict:
        """'NAME,key=value,key=value' -> {'key': 'value', ...}"""
        return dict(part.split("=", 1) for part in response.split(",")[1:] if "=" in part)

    def stop(self) -> None:
        self._send("STOP")

    # ------------------------------------------------ forward: closed-loop --

    def _drive_straight_cm(self, cm: int) -> None:
        self._send(f"STRAIGHT,GOCM,{cm}")
        self._poll_until_done("STRAIGHT,STATUS")

    def _turn_arc(self, radius_cm: float, signed_degrees: int) -> None:
        radius_mm = round(radius_cm * 10)
        self._send(f"TURN,START,R={radius_mm},A={signed_degrees}")
        self._poll_until_done("TURN,STATUS")

    def _poll_until_done(self, status_command: str) -> None:
        deadline = time.monotonic() + COMMAND_TIMEOUT_S
        while True:
            fields = self._parse_fields(self._send(status_command))
            state = fields.get("state")
            if state == "DONE":
                return
            if state in ("ABORTED", "TIMEOUT"):
                raise MotorControllerError(f"{status_command} reported {state}: {fields}")
            if time.monotonic() > deadline:
                self.stop()
                raise MotorControllerError(f"{status_command} never reached DONE within {COMMAND_TIMEOUT_S}s")
            time.sleep(POLL_INTERVAL_S)

    # ---------------------------------------------------- raw fallback path --

    def _set_steering(self, turn: str) -> None:
        if turn == "straight":
            self._send(f"STEER,US,{SERVO_CENTER_US}")
        elif turn == "left":
            self._send(f"STEER,US,{RAW_STEER_LEFT_US}")
        elif turn == "right":
            self._send(f"STEER,US,{RAW_STEER_RIGHT_US}")
        else:
            raise ValueError(f"unknown turn value: {turn!r} (expected 'left', 'right', or 'straight')")

    def _distance_travelled_cm(self) -> float:
        """Distance covered since the last ENC,RESET, always >= 0 regardless
        of whether the robot is driving forward or backward."""
        fields = self._parse_fields(self._send("ENC,GET,CM"))
        left_cm, right_cm = float(fields["l_cm"]), float(fields["r_cm"])
        return abs(left_cm + right_cm) / 2

    def _execute_raw(self, command: dict) -> None:
        """Open-loop fallback: fixed steering + poll encoder distance, no
        gyro correction. Used for backward commands and for forward
        commands too short/long/shallow for STRAIGHT,GOCM/TURN,START."""
        self._set_steering(command["turn"])
        self._send("ENC,RESET")

        pwm = RAW_DRIVE_PWM if command["direction"] == "forward" else -RAW_DRIVE_PWM
        self._send(f"MOTOR,B,{pwm},{pwm}")

        target_cm = command["distance_cm"]
        deadline = time.monotonic() + COMMAND_TIMEOUT_S
        try:
            while self._distance_travelled_cm() < target_cm - DISTANCE_TOLERANCE_CM:
                if time.monotonic() > deadline:
                    raise MotorControllerError(
                        f"raw command did not reach {target_cm}cm within {COMMAND_TIMEOUT_S}s "
                        "(wheel stuck? RAW_DRIVE_PWM too low? check hardware)"
                    )
                time.sleep(POLL_INTERVAL_S)
        finally:
            self.stop()  # always stop, even if the command timed out or raised

    # -------------------------------------------------------- public API --

    def execute_command(self, command: dict) -> None:
        """Drive one command from planner.plan_mission()'s `legs[].commands`:
        {"direction": "forward"|"backward", "turn": "left"|"right"|"straight",
         "distance_cm": float, "degrees": float}."""
        direction = command["direction"]
        turn = command["turn"]

        if direction == "forward" and turn == "straight":
            cm = round(command["distance_cm"])
            if STRAIGHT_MIN_CM <= cm <= STRAIGHT_MAX_CM:
                self._drive_straight_cm(cm)
                return
        elif direction == "forward":
            degrees = round(command["degrees"])
            if degrees >= TURN_MIN_ANGLE_DEG:
                signed_degrees = degrees if turn == "right" else -degrees
                self._turn_arc(TURNING_RADIUS_CM, signed_degrees)
                return

        self._execute_raw(command)

    def execute_leg(self, commands: list[dict]) -> None:
        """Drive every command in one leg, in order, back to back."""
        for command in commands:
            self.execute_command(command)
