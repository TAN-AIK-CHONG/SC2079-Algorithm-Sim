import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

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

# Fallback viewing-pose nudges (in grid cells), tried in order when an
# obstacle's ideal viewing pose is blocked - a neighbouring obstacle sitting
# in it, or the obstacle close enough to a wall that the ideal standoff puts
# the robot out of bounds. (0, 0) is the ideal itself. Column 1 is toward
# the obstacle: it eats into the 20cm camera clearance, capped at
# CAMERA_CLEARANCE_CELLS - 1 so at least 10cm is kept and the robot
# footprint never reaches the obstacle. Column 2 is sideways along the face:
# the image drifts off-centre in frame, capped at 2 cells / 20cm. The
# single step back (-1, 0) is a last resort for a neighbour clipping the
# ideal pose from behind.
VIEWING_POSE_OFFSET_CELLS = (
    (0, 0),
    (0, 1), (0, -1),
    (1, 0),
    (1, 1), (1, -1),
    (0, 2), (0, -2),
    (-1, 0),
)

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

        Returned clockwise (origin -> forward -> forward+right -> right),
        starting from the robot's own origin. Nothing downstream cares about
        winding direction (collision.py's SAT test only needs two adjacent
        edges, in either sense) - this is purely a note for readers, since
        Obstacle.footprint_corners_cm() below is genuinely anticlockwise
        despite the similar-sounding docstring, and it's easy to assume the
        two match.
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

    def candidate_viewing_poses(self) -> list[Robot]:
        """cm_viewing_position() (the ideal) first, then fallback poses for
        when it's blocked - see VIEWING_POSE_OFFSET_CELLS. Every candidate
        keeps the ideal facing (the camera still has to point at the image);
        only the standing position shifts. The caller walks these in order
        and takes the first collision-free one (algorithms/graph.py's
        _resolve_viewing_pose)."""
        grid_x, grid_y, facing = self.grid_viewing_position()
        toward_x, toward_y = facing.value  # unit vector from the robot toward the image
        along_x, along_y = -toward_y, toward_x  # perpendicular, i.e. along the face
        return [
            Robot.from_grid(
                grid_x + toward_x * toward + along_x * along,
                grid_y + toward_y * toward + along_y * along,
                facing,
            )
            for toward, along in VIEWING_POSE_OFFSET_CELLS
        ]


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
