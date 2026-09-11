import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

ARENA_LENGTH_CM = 200
ROBOT_LENGTH_CM = 23
ROBOT_WIDTH_CM = 19

OBSTACLE_FOOTPRINT_LENGTH_CM = 10
OBSTACLE_MARGIN_CM = 5
CAMERA_CLEARANCE_LENGTH_CM = 20

NUM_GRIDS = 20
GRID_LENGTH_CM = ARENA_LENGTH_CM // NUM_GRIDS
OBSTACLE_FOOTPRINT_CELLS = OBSTACLE_FOOTPRINT_LENGTH_CM // GRID_LENGTH_CM

Point = tuple[float, float]
Corners = tuple[Point, Point, Point, Point]  # a footprint outline, walked in order


class Direction(Enum):
    """The four axis-aligned headings, as (dx, dy) unit vectors."""

    NORTH = (0, 1)
    SOUTH = (0, -1)
    EAST = (1, 0)
    WEST = (-1, 0)

    @property
    def theta_rad(self) -> float:
        """A facing is just a heading restricted to four values."""
        dx, dy = self.value
        return math.atan2(dy, dx)

    @property
    def opposite(self) -> "Direction":
        dx, dy = self.value
        return Direction((-dx, -dy))


@dataclass
class MotionPrimitive:
    name: Literal[
        "forward_straight",
        "forward_left",
        "forward_right",
        "reverse_straight",
        "reverse_left",
        "reverse_right",
    ]
    direction: int
    dtheta: float
    distance: int


@dataclass(frozen=True)
class Robot:
    x_cm: float  # rear-axle midpoint
    y_cm: float  # rear-axle midpoint
    theta_rad: float

    @classmethod
    def from_grid(cls, x_coord: int, y_coord: int, facing: Direction) -> "Robot":
        """Builds Robot starting position from grid coordinates"""
        corners = cls(0.0, 0.0, facing.theta_rad).footprint_corners_cm()
        return cls(
            x_coord * GRID_LENGTH_CM - min(cx for cx, _ in corners),
            y_coord * GRID_LENGTH_CM - min(cy for _, cy in corners),
            facing.theta_rad,
        )

    def footprint_corners_cm(self) -> Corners:
        """
        The four corners of the robot's footprint at this pose, in cm.

        Returned clockwise, starting from left-rear wheel.
        """
        forward_x, forward_y = math.cos(self.theta_rad), math.sin(self.theta_rad)
        right_x, right_y = forward_y, -forward_x
        half = ROBOT_WIDTH_CM / 2

        def corner(along: float, across: float) -> Point:
            return (
                self.x_cm + along * forward_x + across * right_x,
                self.y_cm + along * forward_y + across * right_y,
            )

        return (
            corner(0.0, -half),
            corner(ROBOT_LENGTH_CM, -half),
            corner(ROBOT_LENGTH_CM, half),
            corner(0.0, half),
        )


@dataclass(frozen=True)
class Obstacle:
    id: int
    x_coord: int
    y_coord: int
    image_side: Direction

    def footprint_corners_cm(self, margin_cm: float = 0.0) -> Corners:
        """
        The four corners of the obstacle's square footprint, in cm, grown by
        `margin_cm` on every side.

        Returned anticlockwise from the bottom-left, the cell's own origin.
        """
        min_x = self.x_coord * GRID_LENGTH_CM - margin_cm
        min_y = self.y_coord * GRID_LENGTH_CM - margin_cm
        max_x = min_x + OBSTACLE_FOOTPRINT_LENGTH_CM + 2 * margin_cm
        max_y = min_y + OBSTACLE_FOOTPRINT_LENGTH_CM + 2 * margin_cm
        return (
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        )

    def inflated_footprint_corners_cm(self) -> Corners:
        return self.footprint_corners_cm(OBSTACLE_MARGIN_CM)

    def centre_cm(self) -> Point:
        half = OBSTACLE_FOOTPRINT_LENGTH_CM / 2
        return (
            self.x_coord * GRID_LENGTH_CM + half,
            self.y_coord * GRID_LENGTH_CM + half,
        )

    def cm_viewing_position(self) -> Robot:
        dx, dy = self.image_side.value
        standoff = (
            OBSTACLE_FOOTPRINT_LENGTH_CM / 2
            + CAMERA_CLEARANCE_LENGTH_CM
            + ROBOT_LENGTH_CM
        )
        centre_x, centre_y = self.centre_cm()
        return Robot(
            centre_x + dx * standoff,
            centre_y + dy * standoff,
            self.image_side.opposite.theta_rad,  # look back at the image face
        )


def parse_scenario(data: dict) -> tuple[Robot, list[Obstacle]]:
    """Build a Robot + Obstacle list from the same JSON shape as the generated maps
    (used by both the map tooling and the live API, so the two never drift).

    The JSON speaks grid cells and named facings; everything past this point speaks
    cm and radians.
    """
    robot = Robot.from_grid(
        data["robot"]["x_coord"],
        data["robot"]["y_coord"],
        Direction[data["robot"]["facing"]],
    )

    obstacles = [
        Obstacle(
            obs["id"],
            obs["x_coord"],
            obs["y_coord"],
            Direction[obs["image_side"]],
        )
        for obs in data["obstacles"]
    ]

    return robot, obstacles
