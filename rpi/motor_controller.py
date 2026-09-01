"""
Runs ON THE RPI - translates the drive commands produced by src/planner.py
into the STM32 serial protocol. Called directly by main.py; there is no
network/API step, everything runs in one process.

"""

from __future__ import annotations

import os
import time

import serial
import serial.tools.list_ports

from algorithms.hybrid_astar import LEFT_TURNING_RADIUS_CM, RIGHT_TURNING_RADIUS_CM  # single source of truth
# - the radii the paths were actually planned at (see src/algorithms/hybrid_astar.py's
# _motion_primitives). Must send the STM the same radius the arc was planned with, or the
# physical robot's arc won't match the planned path.


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

SERVO_CENTER_US = 1712

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

    def _turn_arc(self, turn: str, radius_cm: float, signed_degrees: int) -> None:
        self._send(f"TURN,{turn},R={round(radius_cm)},A={signed_degrees}")
        self._poll_until_done("TURN,STATUS")

    def _poll_until_done(self, status_command: str) -> None:
        mode_stop_command = status_command.replace(",STATUS", ",STOP")
        deadline = time.monotonic() + COMMAND_TIMEOUT_S
        while True:
            fields = self._parse_fields(self._send(status_command))
            state = fields.get("state")
            if state == "DONE":
                return
            # Anything that isn't actively progressing is a failure - not
            # just ABORTED/TIMEOUT. TURN,STATUS in particular can also
            # report WRONG_SIGN/NO_TURN/BAD_CONFIG/NO_IMU (test_commands.c's
            # TurnStateText); hardcoding only the two old names would leave
            # us silently polling those until our own COMMAND_TIMEOUT_S
            # instead of failing fast with the STM's actual reason.
            if state not in ("RUNNING", "IDLE"):
                raise MotorControllerError(f"{status_command} reported {state}: {fields}")
            if time.monotonic() > deadline:
                self._send(mode_stop_command)
                raise MotorControllerError(f"{status_command} never reached DONE within {COMMAND_TIMEOUT_S}s")
            time.sleep(POLL_INTERVAL_S)

    # ---------------------------------------------------- raw fallback path --

    def _set_steering(self, turn: str) -> None:
        if turn == "STRAIGHT":
            self._send(f"STEER,US,{SERVO_CENTER_US}")
        elif turn == "LEFT":
            self._send(f"STEER,US,{RAW_STEER_LEFT_US}")
        elif turn == "RIGHT":
            self._send(f"STEER,US,{RAW_STEER_RIGHT_US}")
        else:
            raise ValueError(f"unknown turn value: {turn!r} (expected 'LEFT', 'RIGHT', or 'STRAIGHT')")

    def _distance_travelled_cm(self) -> float:
        """Distance covered since the last ENC,RESET, always >= 0 regardless
        of whether the robot is driving forward or backward."""
        fields = self._parse_fields(self._send("ENC,GET,CM"))
        left_cm, right_cm = float(fields["l_cm"]), float(fields["r_cm"])
        return abs(left_cm + right_cm) / 2

    def _execute_raw(self, command) -> None:
        """Open-loop fallback: fixed steering + poll encoder distance, no
        gyro correction. Used for backward commands and for forward
        commands too short/long/shallow for STRAIGHT,GOCM/TURN,START."""
        self._set_steering(command.turn)
        self._send("ENC,RESET")

        pwm = RAW_DRIVE_PWM if command.direction == "FORWARD" else -RAW_DRIVE_PWM
        self._send(f"MOTOR,B,{pwm},{pwm}")

        target_cm = command.distance_cm
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

    def execute_command(self, command) -> None:
        """Drive one command from a planner.Leg's `.commands` list - a
        planner.Command dataclass: direction "FORWARD"|"REVERSE", turn
        "STRAIGHT"|"LEFT"|"RIGHT", distance_cm: int, swept_angle_deg: int|None
        (None for STRAIGHT commands)."""
        direction = command.direction
        turn = command.turn

        if turn == "STRAIGHT":
            if direction == "FORWARD":
                cm = round(command.distance_cm)
                if STRAIGHT_MIN_CM <= cm <= STRAIGHT_MAX_CM:
                    self._drive_straight_cm(cm)
                    return
            # REVERSE STRAIGHT has no closed-loop equivalent yet - falls
            # through to the raw fallback below.
        else:
            degrees = round(command.swept_angle_deg)
            if degrees >= TURN_MIN_ANGLE_DEG:
                signed_degrees = degrees if direction == "FORWARD" else -degrees
                radius_cm = LEFT_TURNING_RADIUS_CM if turn == "LEFT" else RIGHT_TURNING_RADIUS_CM
                self._turn_arc(turn, radius_cm, signed_degrees)
                return

        self._execute_raw(command)

    def execute_leg(self, commands: list) -> None:
        """Drive every command in one leg, in order, back to back."""
        for command in commands:
            self.execute_command(command)
