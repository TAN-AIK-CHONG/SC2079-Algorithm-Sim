"""
Tests motor_controller.py's logic against a *fake* serial port (no real
STM32 needed) - verifies the command sequence and routing (STRAIGHT,GOCM /
TURN,START vs raw fallback) are correct, independent of whatever the real
calibration constants turn out to be. Run with: pytest test_motor_controller.py
(from inside rpi/)
"""

import motor_controller as mc
import pytest


class FakeSerial:
    """Stands in for pyserial's Serial. Records every command sent, and
    simulates the STM's STRAIGHT/TURN state machines: RUNNING for
    `polls_until_done` STATUS polls, then `done_state` (default DONE).
    Also fakes ENC,GET,CM for the raw fallback path, same as before."""

    def __init__(self, *args, **kwargs):
        self.sent: list[str] = []
        self._distance_cm = 0.0
        self._cm_per_poll = 1.0
        self._driving = False
        self.polls_until_done = 1
        self.done_state = "DONE"
        self._status_polls = 0

    def write(self, data: bytes) -> None:
        command = data.decode("ascii").strip()
        self.sent.append(command)

        if command == "ENC,RESET":
            self._distance_cm = 0.0
        elif command.startswith("MOTOR,B,"):
            self._driving = True
        elif command == "STOP":
            self._driving = False
        elif command == "ENC,GET,CM" and self._driving:
            self._distance_cm += self._cm_per_poll
        elif command.startswith("STRAIGHT,GOCM,") or command.startswith("TURN,START,"):
            self._status_polls = 0

        self._pending_response = self._respond_to(command)

    def _respond_to(self, command: str) -> str:
        if command == "HELLO":
            return "HELLO,mdp21_v8"
        if command == "ENC,GET,CM":
            return f"ENC,l_cm={self._distance_cm:.3f},r_cm={self._distance_cm:.3f}"
        if command.startswith("STRAIGHT,GOCM,"):
            return f"OK,target_cm={command.split(',')[2]}"
        if command == "STRAIGHT,STATUS":
            self._status_polls += 1
            state = self.done_state if self._status_polls >= self.polls_until_done else "RUNNING"
            return f"STRAIGHT,state={state},elapsed=100,target_cm=10.000,l_cm=10.000,r_cm=10.000,diff_cm=0.000,yaw=0.000"
        if command.startswith("TURN,START,"):
            parts = dict(p.split("=", 1) for p in command.split(",")[2:])
            return f"OK,r={parts['R']},a={parts['A']}"
        if command == "TURN,STATUS":
            self._status_polls += 1
            state = self.done_state if self._status_polls >= self.polls_until_done else "RUNNING"
            return f"TURN,state={state},elapsed=100,r=250,target_deg=30.000,yaw=0.000"
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


def test_forward_straight_uses_straight_gocm(controller):
    controller._ser.sent.clear()
    controller.execute_command({"direction": "forward", "turn": "straight", "distance_cm": 30.0, "degrees": 0.0})

    assert controller._ser.sent[0] == "STRAIGHT,GOCM,30"
    assert "STRAIGHT,STATUS" in controller._ser.sent
    # never touches the raw fallback commands
    assert not any(c.startswith("MOTOR,B,") for c in controller._ser.sent)


def test_forward_right_turn_uses_turn_start_with_positive_angle(controller):
    controller._ser.sent.clear()
    controller.execute_command({"direction": "forward", "turn": "right", "distance_cm": 15.0, "degrees": 30.0})
    assert controller._ser.sent[0] == "TURN,START,R=250,A=30"


def test_forward_left_turn_uses_turn_start_with_negative_angle(controller):
    controller._ser.sent.clear()
    controller.execute_command({"direction": "forward", "turn": "left", "distance_cm": 15.0, "degrees": 30.0})
    assert controller._ser.sent[0] == "TURN,START,R=250,A=-30"


def test_backward_command_falls_back_to_raw_motor(controller):
    # STRAIGHT,GOCM/TURN,START are forward-only (see drive_control.h) -
    # backward must go through raw MOTOR,B with negative PWM.
    controller.execute_command({"direction": "backward", "turn": "straight", "distance_cm": 20.0, "degrees": 0.0})
    motor_cmds = [c for c in controller._ser.sent if c.startswith("MOTOR,B,")]
    assert motor_cmds == [f"MOTOR,B,{-mc.RAW_DRIVE_PWM},{-mc.RAW_DRIVE_PWM}"]
    assert controller._ser.sent[-1] == "STOP"


def test_short_straight_segment_falls_back_to_raw(controller):
    # STRAIGHT,GOCM only accepts 10-500cm; below that it would be rejected.
    controller.execute_command({"direction": "forward", "turn": "straight", "distance_cm": 4.0, "degrees": 0.0})
    assert not any(c.startswith("STRAIGHT,GOCM,") for c in controller._ser.sent)
    assert any(c.startswith("MOTOR,B,") for c in controller._ser.sent)


def test_shallow_turn_falls_back_to_raw(controller):
    # TURN,START only accepts |angle| >= 5deg.
    controller.execute_command({"direction": "forward", "turn": "left", "distance_cm": 2.0, "degrees": 2.0})
    assert not any(c.startswith("TURN,START,") for c in controller._ser.sent)
    assert any(c.startswith("MOTOR,B,") for c in controller._ser.sent)
    assert any(c == f"STEER,US,{mc.RAW_STEER_LEFT_US}" for c in controller._ser.sent)


def test_execute_leg_runs_every_command_in_order(controller):
    commands = [
        {"direction": "forward", "turn": "right", "distance_cm": 15.0, "degrees": 30.0},
        {"direction": "forward", "turn": "straight", "distance_cm": 20.0, "degrees": 0.0},
    ]
    controller._ser.sent.clear()
    controller.execute_leg(commands)

    assert controller._ser.sent[0] == "TURN,START,R=250,A=30"
    assert "STRAIGHT,GOCM,20" in controller._ser.sent


def test_straight_aborted_raises(controller):
    controller._ser.done_state = "ABORTED"
    with pytest.raises(mc.MotorControllerError, match="ABORTED"):
        controller.execute_command({"direction": "forward", "turn": "straight", "distance_cm": 30.0, "degrees": 0.0})


def test_turn_never_completing_times_out_and_stops(controller, monkeypatch):
    monkeypatch.setattr(mc, "COMMAND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mc, "POLL_INTERVAL_S", 0.01)
    controller._ser.polls_until_done = 10_000  # never reaches "done" in time

    with pytest.raises(mc.MotorControllerError, match="never reached DONE"):
        controller.execute_command({"direction": "forward", "turn": "right", "distance_cm": 15.0, "degrees": 30.0})

    assert controller._ser.sent[-1] == "STOP"


def test_raw_command_times_out_if_target_never_reached(controller, monkeypatch):
    monkeypatch.setattr(mc, "COMMAND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mc, "POLL_INTERVAL_S", 0.01)
    controller._ser._cm_per_poll = 0.0  # simulate a stuck/stalled robot

    with pytest.raises(mc.MotorControllerError, match="did not reach"):
        controller.execute_command({"direction": "backward", "turn": "straight", "distance_cm": 100.0, "degrees": 0.0})

    assert controller._ser.sent[-1] == "STOP"
