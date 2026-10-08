import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

ARENA_LENGTH_CM = 200
ROBOT_LENGTH_CM = 23
ROBOT_WIDTH_CM = 19
# The pose is the rear-axle midpoint (what the STM drives), not the back edge.
AXLE_TO_REAR_CM = 4
AXLE_TO_FRONT_CM = ROBOT_LENGTH_CM - AXLE_TO_REAR_CM

OBSTACLE_FOOTPRINT_LENGTH_CM = 10
OBSTACLE_MARGIN_CM = 5
CAMERA_CLEARANCE_LENGTH_CM = 20

NUM_GRIDS = 20
GRID_LENGTH_CM = ARENA_LENGTH_CM // NUM_GRIDS
OBSTACLE_FOOTPRINT_CELLS = OBSTACLE_FOOTPRINT_LENGTH_CM // GRID_LENGTH_CM

# A pose counts as "arrived" at an obstacle anywhere the camera can
# photograph its image (see Obstacle.is_viewed_from), not only at the one
# ideal cm_viewing_position, so the robot never shuffles on the spot to
# square up exactly. The camera sits on the nose.
MIN_CAMERA_DISTANCE_CM = 20
MAX_CAMERA_DISTANCE_CM = 50
# Images stay recognisable when viewed a bit obliquely.
MAX_VIEWING_ANGLE_RAD = math.radians(10)
# Keeps the image inside the camera frame.
MAX_HEADING_ERROR_RAD = math.radians(10)

Point = tuple[float, float]
Corners = tuple[Point, Point, Point, Point]  # a footprint outline, walked in order


def _angle_between(a: Point, b: Point) -> float:
    """Unsigned angle between two vectors, in radians (0 to pi)."""
    cross = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return abs(math.atan2(cross, dot))


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

    def heading_vector(self) -> Point:
        return math.cos(self.theta_rad), math.sin(self.theta_rad)

    def camera_position_cm(self) -> Point:
        """The camera sits on the nose, AXLE_TO_FRONT_CM ahead of the rear axle."""
        forward_x, forward_y = self.heading_vector()
        return (
            self.x_cm + AXLE_TO_FRONT_CM * forward_x,
            self.y_cm + AXLE_TO_FRONT_CM * forward_y,
        )

    def footprint_corners_cm(self) -> Corners:
        """
        The four corners of the robot's footprint at this pose, in cm.

        Returned clockwise, starting from left-rear wheel.
        """
        forward_x, forward_y = self.heading_vector()
        right_x, right_y = forward_y, -forward_x
        half = ROBOT_WIDTH_CM / 2

        def corner(along: float, across: float) -> Point:
            return (
                self.x_cm + along * forward_x + across * right_x,
                self.y_cm + along * forward_y + across * right_y,
            )

        rear = -AXLE_TO_REAR_CM
        front = ROBOT_LENGTH_CM - AXLE_TO_REAR_CM
        return (
            corner(rear, -half),
            corner(front, -half),
            corner(front, half),
            corner(rear, half),
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
            + AXLE_TO_FRONT_CM   # place the rear AXLE (what the STM drives), so the
                                 # nose ends CAMERA_CLEARANCE_LENGTH_CM off the face
        )
        centre_x, centre_y = self.centre_cm()
        return Robot(
            centre_x + dx * standoff,
            centre_y + dy * standoff,
            self.image_side.opposite.theta_rad,  # look back at the image face
        )

    def image_centre_cm(self) -> Point:
        centre_x, centre_y = self.centre_cm()
        normal_x, normal_y = self.image_side.value
        half = OBSTACLE_FOOTPRINT_LENGTH_CM / 2
        return centre_x + normal_x * half, centre_y + normal_y * half

    def is_viewed_from(self, robot: Robot) -> bool:
        """Whether the camera at this pose can photograph the image: within
        range of it, standing roughly in front of the face rather than off
        to the side, and pointing at it."""
        camera_x, camera_y = robot.camera_position_cm()
        image_x, image_y = self.image_centre_cm()
        image_to_camera = (camera_x - image_x, camera_y - image_y)
        camera_to_image = (-image_to_camera[0], -image_to_camera[1])

        distance_cm = math.hypot(*image_to_camera)
        is_in_range = MIN_CAMERA_DISTANCE_CM <= distance_cm <= MAX_CAMERA_DISTANCE_CM
        is_in_front = _angle_between(self.image_side.value, image_to_camera) <= MAX_VIEWING_ANGLE_RAD
        is_facing_image = _angle_between(robot.heading_vector(), camera_to_image) <= MAX_HEADING_ERROR_RAD
        return is_in_range and is_in_front and is_facing_image


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
