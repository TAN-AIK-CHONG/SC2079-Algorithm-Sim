"""
Tests motor_controller.py's logic against a *fake* serial port (no real
STM32 needed) - verifies the command sequence and stopping condition are
correct, independent of whatever the real calibration constants turn out to
be. Run with: pytest test_motor_controller.py (from inside rpi/)
"""

import motor_controller as mc
import pytest


class FakeSerial:
    """Stands in for pyserial's Serial. Records every command sent, and
    fakes ENC,GET,CM readings that increase by `cm_per_poll` each time
    ENC,GET,CM is polled after a MOTOR,B command - simulating a robot that
    is actually moving."""

    def __init__(self, *args, **kwargs):
        self.sent: list[str] = []
        self._distance_cm = 0.0
        self._cm_per_poll = 1.0
        self._driving = False

    def write(self, data: bytes) -> None:
        command = data.decode("ascii").strip()
        self.sent.append(command)

        if command == "ENC,RESET":
            self._distance_cm = 0.0
        elif command.startswith("MOTOR,B,"):
            self._driving = True
        elif command == "STOP":
            self._driving = False
        elif command.startswith("ENC,GET,CM") and self._driving:
            self._distance_cm += self._cm_per_poll

        self._pending_response = self._respond_to(command)

    def _respond_to(self, command: str) -> str:
        if command == "HELLO":
            return "HELLO,mdp21_v8"
        if command == "ENC,GET,CM":
            return f"ENC,l_cm={self._distance_cm:.3f},r_cm={self._distance_cm:.3f}"
        return "OK"

    def readline(self) -> bytes:
        return (self._pending_response + "\r\n").encode("ascii")

    def close(self) -> None:
        pass


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(mc.serial, "Serial", FakeSerial)
    ctrl = mc.MotorController()
    return ctrl


def test_handshake_happens_on_connect(controller):
    assert controller._ser.sent == ["HELLO"]


def test_forward_straight_command_sequence(controller):
    controller._ser.sent.clear()
    controller.execute_command({"direction": "forward", "turn": "straight", "distance_cm": 3.0, "degrees": 0.0})

    assert controller._ser.sent[0] == f"STEER,US,{mc.STEER_CENTER_US}"
    assert controller._ser.sent[1] == "ENC,RESET"
    assert controller._ser.sent[2] == f"MOTOR,B,{mc.DRIVE_PWM},{mc.DRIVE_PWM}"
    assert controller._ser.sent[-1] == "STOP"


def test_left_turn_uses_left_steering_value(controller):
    controller._ser.sent.clear()
    controller.execute_command({"direction": "forward", "turn": "left", "distance_cm": 2.0, "degrees": 45.0})
    assert controller._ser.sent[0] == f"STEER,US,{mc.STEER_LEFT_US}"


def test_right_turn_uses_right_steering_value(controller):
    controller._ser.sent.clear()
    controller.execute_command({"direction": "forward", "turn": "right", "distance_cm": 2.0, "degrees": 45.0})
    assert controller._ser.sent[0] == f"STEER,US,{mc.STEER_RIGHT_US}"


def test_backward_command_still_stops_at_target_distance(controller):
    # Regression test: encoders are "forward=positive" per the protocol doc,
    # so a naive (non-abs) distance check would never terminate for backward
    # moves. This confirms it does terminate, using MOTOR,B with negative PWM.
    controller.execute_command({"direction": "backward", "turn": "straight", "distance_cm": 3.0, "degrees": 0.0})
    motor_cmds = [c for c in controller._ser.sent if c.startswith("MOTOR,B,")]
    assert motor_cmds == [f"MOTOR,B,{-mc.DRIVE_PWM},{-mc.DRIVE_PWM}"]
    assert controller._ser.sent[-1] == "STOP"


def test_execute_leg_runs_every_command_in_order(controller):
    commands = [
        {"direction": "forward", "turn": "right", "distance_cm": 2.0, "degrees": 30.0},
        {"direction": "forward", "turn": "straight", "distance_cm": 3.0, "degrees": 0.0},
    ]
    controller._ser.sent.clear()
    controller.execute_leg(commands)

    steer_commands = [c for c in controller._ser.sent if c.startswith("STEER,US,")]
    assert steer_commands == [f"STEER,US,{mc.STEER_RIGHT_US}", f"STEER,US,{mc.STEER_CENTER_US}"]


def test_command_times_out_if_target_never_reached(controller, monkeypatch):
    monkeypatch.setattr(mc, "COMMAND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mc, "POLL_INTERVAL_S", 0.01)
    controller._ser._cm_per_poll = 0.0  # simulate a stuck/stalled robot

    with pytest.raises(mc.MotorControllerError, match="did not reach"):
        controller.execute_command({"direction": "forward", "turn": "straight", "distance_cm": 100.0, "degrees": 0.0})

    # Must still have sent STOP despite raising - safety net in the `finally`.
    assert controller._ser.sent[-1] == "STOP"
