import math
from dataclasses import dataclass
from enum import Enum

ARENA_LENGTH_CM = 200
ROBOT_FOOTPRINT_LENGTH_CM = 30
OBSTACLE_FOOTPRINT_LENGTH_CM = 10
CAMERA_CLEARANCE_LENGTH_CM = 20

NUM_GRIDS = 20
GRID_LENGTH_CM = ARENA_LENGTH_CM // NUM_GRIDS

ROBOT_FOOTPRINT_CELLS = ROBOT_FOOTPRINT_LENGTH_CM // GRID_LENGTH_CM
OBSTACLE_FOOTPRINT_CELLS = OBSTACLE_FOOTPRINT_LENGTH_CM // GRID_LENGTH_CM

CAMERA_CLEARANCE_CELLS = CAMERA_CLEARANCE_LENGTH_CM // GRID_LENGTH_CM

DEPTH_CLEARANCE_CELLS = ROBOT_FOOTPRINT_CELLS + CAMERA_CLEARANCE_CELLS
ALIGNMENT_OFFSET_CELLS = (ROBOT_FOOTPRINT_CELLS - OBSTACLE_FOOTPRINT_CELLS) // 2

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


@dataclass(frozen=True)
class Robot:
    x_cm: float
    y_cm: float
    theta_rad: float

    @classmethod
    def from_grid(cls, x_coord: int, y_coord: int, facing: Direction) -> "Robot":
        return cls(
            x_coord * GRID_LENGTH_CM,
            y_coord * GRID_LENGTH_CM,
            facing.theta_rad,
        )

    def footprint_corners_cm(self) -> Corners:
        """
        The four corners of the robot's square footprint at this pose, in cm.

        Returned anticlockwise from the bottom-left, the robot's own origin.
        """
        forward_x, forward_y = math.cos(self.theta_rad), math.sin(self.theta_rad)
        right_x, right_y = forward_y, -forward_x
        side = ROBOT_FOOTPRINT_LENGTH_CM
        return (
            (self.x_cm, self.y_cm),
            (self.x_cm + side * forward_x, self.y_cm + side * forward_y),
            (
                self.x_cm + side * (forward_x + right_x),
                self.y_cm + side * (forward_y + right_y),
            ),
            (self.x_cm + side * right_x, self.y_cm + side * right_y),
        )


@dataclass(frozen=True)
class Obstacle:
    id: int
    x_coord: int
    y_coord: int
    image_side: Direction

    def footprint_corners_cm(self) -> Corners:
        """
        The four corners of the obstacle's square footprint, in cm.

        Returned anticlockwise from the bottom-left, the cell's own origin.
        """
        min_x = self.x_coord * GRID_LENGTH_CM
        min_y = self.y_coord * GRID_LENGTH_CM
        max_x = min_x + OBSTACLE_FOOTPRINT_LENGTH_CM
        max_y = min_y + OBSTACLE_FOOTPRINT_LENGTH_CM
        return (
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        )

    def grid_viewing_position(self) -> tuple[int, int, Direction]:
        if self.image_side is Direction.SOUTH:
            return (
                self.x_coord - ALIGNMENT_OFFSET_CELLS,
                self.y_coord - DEPTH_CLEARANCE_CELLS,
                Direction.NORTH,
            )
        elif self.image_side is Direction.NORTH:
            return (
                self.x_coord + ALIGNMENT_OFFSET_CELLS + OBSTACLE_FOOTPRINT_CELLS,
                self.y_coord + OBSTACLE_FOOTPRINT_CELLS + DEPTH_CLEARANCE_CELLS,
                Direction.SOUTH,
            )
        elif self.image_side is Direction.WEST:
            return (
                self.x_coord - DEPTH_CLEARANCE_CELLS,
                self.y_coord + ALIGNMENT_OFFSET_CELLS + OBSTACLE_FOOTPRINT_CELLS,
                Direction.EAST,
            )
        else:
            return (
                self.x_coord + OBSTACLE_FOOTPRINT_CELLS + DEPTH_CLEARANCE_CELLS,
                self.y_coord - ALIGNMENT_OFFSET_CELLS,
                Direction.WEST,
            )

    def cm_viewing_position(self) -> Robot:
        return Robot.from_grid(*self.grid_viewing_position())


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
