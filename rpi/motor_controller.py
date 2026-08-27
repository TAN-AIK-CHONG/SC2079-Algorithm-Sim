"""
Runs ON THE RPI - translates the drive commands produced by src/planner.py
into the STM32 serial protocol described in PROTOCOL_en.md. Called directly
by main.py; there is no network/API step, everything runs in one process.

Why no gyro/yaw or STEER,ANGLE: those aren't implemented on the STM yet
(v1.5/v2/v3 per the protocol doc's phase table - only v1 is real today).
Instead, every command - straight or turning - is executed the same way:
set a steering angle, drive, and watch the wheel encoders (already
implemented and calibrated in cm, v1) until `distance_cm` is covered.
That's enough because a turn command's `distance_cm` is already the *arc
length* to travel (see src/algorithms/dubins.py: path_commands), and arc
length is exactly what the encoders measure, regardless of whether the
wheels are pointed straight or turned.

Two constants below MUST be measured on the real robot before this will
drive correctly - the algorithm module has no way to know them:
  - DRIVE_PWM: a raw MOTOR,B PWM value that gives a sensible, safe speed
  - STEER_LEFT_US / STEER_RIGHT_US: raw servo pulse widths that produce a
    turn at TURNING_RADIUS_CM (the radius the paths were planned for -
    see src/algorithms/dubins.py) - not just "some turn", THAT radius,
    otherwise the robot won't end up where the plan expects.
"""

from __future__ import annotations

import time

import serial

SERIAL_PORT = "COM5"  # placeholder - update to whatever port the CH9102F / RPi UART shows up as
BAUD_RATE = 115200

# --- MUST CALIBRATE on the real robot before trusting this to drive -------
DRIVE_PWM = 3500          # TODO: measure a safe, controllable cruising PWM
STEER_CENTER_US = 1712     # protocol doc's default center candidate
STEER_LEFT_US = 1400        # TODO: measure - must yield TURNING_RADIUS_CM, not just "some" left turn
STEER_RIGHT_US = 2000        # TODO: measure - must yield TURNING_RADIUS_CM, not just "some" right turn
# ---------------------------------------------------------------------------

DISTANCE_TOLERANCE_CM = 0.5   # stop once within this much of the target
POLL_INTERVAL_S = 0.02
COMMAND_TIMEOUT_S = 10.0        # safety net: abort a single command if it never reaches target


class MotorControllerError(RuntimeError):
    pass


class MotorController:
    """One instance per serial connection to the STM32. Use execute_leg() /
    execute_command() to drive the commands the algorithm API returns."""

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

    # --------------------------------------------------------- primitives --

    def stop(self) -> None:
        self._send("STOP")

    def _set_steering(self, turn: str) -> None:
        if turn == "straight":
            self._send(f"STEER,US,{STEER_CENTER_US}")
        elif turn == "left":
            self._send(f"STEER,US,{STEER_LEFT_US}")
        elif turn == "right":
            self._send(f"STEER,US,{STEER_RIGHT_US}")
        else:
            raise ValueError(f"unknown turn value: {turn!r} (expected 'left', 'right', or 'straight')")

    def _distance_travelled_cm(self) -> float:
        """Distance covered since the last ENC,RESET, always >= 0 regardless
        of whether the robot is driving forward or backward."""
        response = self._send("ENC,GET,CM")  # "ENC,l_cm=<f>,r_cm=<f>"
        fields = dict(part.split("=", 1) for part in response.split(",")[1:])
        left_cm, right_cm = float(fields["l_cm"]), float(fields["r_cm"])
        return abs(left_cm + right_cm) / 2

    # -------------------------------------------------------- public API --

    def execute_command(self, command: dict) -> None:
        """Drive one command from planner.plan_mission()'s `legs[].commands`:
        {"direction": "forward"|"backward", "turn": "left"|"right"|"straight",
         "distance_cm": float, "degrees": float}."""
        self._set_steering(command["turn"])
        self._send("ENC,RESET")

        pwm = DRIVE_PWM if command["direction"] == "forward" else -DRIVE_PWM
        self._send(f"MOTOR,B,{pwm},{pwm}")

        target_cm = command["distance_cm"]
        deadline = time.monotonic() + COMMAND_TIMEOUT_S
        try:
            while self._distance_travelled_cm() < target_cm - DISTANCE_TOLERANCE_CM:
                if time.monotonic() > deadline:
                    raise MotorControllerError(
                        f"command did not reach {target_cm}cm within {COMMAND_TIMEOUT_S}s "
                        "(wheel stuck? DRIVE_PWM too low? check hardware)"
                    )
                time.sleep(POLL_INTERVAL_S)
        finally:
            self.stop()  # always stop, even if the command timed out or raised

    def execute_leg(self, commands: list[dict]) -> None:
        """Drive every command in one leg, in order, back to back."""
        for command in commands:
            self.execute_command(command)
