"""
Runs ON THE RPI - the recovery move for when image recognition comes back
with a BULLSEYE instead of a numbered image.

A bullseye means the robot drove the planned leg correctly but the image is
on a different face of the obstacle than the map claimed, so it is parked at
the wrong side. detected_bullseye() returns the drive commands that carry it
around to the NEXT face, ending exactly on that face's viewing pose. The RPi
takes another photo there and calls this again if it gets another bullseye -
three calls cover all four faces:

    controller.execute_leg(detected_bullseye())

Face order is SOUTH -> EAST -> NORTH -> WEST -> SOUTH. Viewed from above with 
north up, starting parked south of the obstacle facing north:

        [N]                     the robot travels S -> E, keeping the
    [W] OBS [E]   robot goes     obstacle on its LEFT, and its own heading
        [S]       S -> E         advances +90 degrees (north -> west).

PRECONDITION: the robot is already at a viewing pose - squared up to the
face, centred on it, front of the 30x30 footprint 20cm off the obstacle.
That is precisely what model.Obstacle.cm_viewing_position() returns, and the
whole maneuver is defined against it. Called from anywhere else it lands
somewhere else.

The maneuver is BLIND: it knows the obstacle it is circling but nothing about
the arena walls or the other obstacles. Replayed over every obstacle/face
pair whose viewing poses are themselves valid, it fits inside the arena 66%
of the time with the obstacle alone, and 29% of the time on the dense random
layouts in src/testing/generated_maps. The caller is responsible for deciding
there is room.
"""

from __future__ import annotations

import math

from algorithms.hybrid_astar import LEFT_TURNING_RADIUS_CM, RIGHT_TURNING_RADIUS_CM
from model import Robot
from planner import Command, _apply_command

# The three arcs, in degrees swept - the cached output of the solver at the
# bottom of this file. SOLVED FOR THE RADII ABOVE: they are not a tuning
# knob, and changing LEFT_TURNING_RADIUS_CM or RIGHT_TURNING_RADIUS_CM
# invalidates them (test_bullseye.py fails loudly if that happens; run
# `python bullseye.py` to re-solve for the new radii and paste the result
# back here).
#
# Why these three: the four viewing poses of an obstacle are exact 90-degree
# rotations of one another about the obstacle's centre, so ONE sequence
# serves all four hops. The signed sweeps must therefore come to exactly
# +90: -73 + 96 + 67. There is no forward-only solution at these radii - the
# turning circles are too wide for the 40cm standoff - so the third arc
# reverses. Among the triples that land on the viewing pose, this one is the
# most accurate (0.25cm) of those keeping essentially the most room the
# family allows around the obstacle (4.9cm of a possible 5.2cm).
FORWARD_RIGHT_DEG = 73
FORWARD_LEFT_DEG = 96
REVERSE_RIGHT_DEG = 67


def _arc(turn: str, direction: str, swept_angle_deg: int) -> Command:
    """One arc Command at the fixed radius motor_controller.py will drive it
    at. distance_cm has to be the arc length for that radius, not an
    independent number: the STM only gets R and A (see
    MotorController._turn_arc), but planner._apply_command reconstructs the
    curve as distance_cm/swept_angle_deg, and the two have to describe the
    same arc or the map tooling draws a path the robot never takes."""
    radius_cm = LEFT_TURNING_RADIUS_CM if turn == "LEFT" else RIGHT_TURNING_RADIUS_CM
    return Command(
        direction=direction,
        turn=turn,
        distance_cm=round(radius_cm * math.radians(swept_angle_deg)),
        swept_angle_deg=swept_angle_deg,
    )


def detected_bullseye() -> list[Command]:
    """Drive commands from the current viewing pose to the next face's
    viewing pose (SOUTH -> EAST -> NORTH -> WEST -> SOUTH).

    Same list[Command] shape as a planner.Leg's `.commands`, so it goes
    straight into MotorController.execute_leg(). ~119cm of driving; every
    command is an arc of at least TURN_MIN_ANGLE_DEG, so all three go out as
    closed-loop TURN commands and none fall through to the open-loop raw
    fallback."""
    return [
        _arc("RIGHT", "FORWARD", FORWARD_RIGHT_DEG),
        _arc("LEFT", "FORWARD", FORWARD_LEFT_DEG),
        _arc("RIGHT", "REVERSE", REVERSE_RIGHT_DEG),
    ]