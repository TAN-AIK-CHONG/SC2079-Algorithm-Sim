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


class Direction(Enum):
    NORTH = (0, 1)
    SOUTH = (0, -1)
    EAST = (1, 0)
    WEST = (-1, 0)


@dataclass(frozen=True)
class Obstacle:
    id: str
    x_coord: int
    y_coord: int
    image_side: Direction

    def desired_robot_position(self) -> tuple[int, int, Direction]:
        if self.image_side is Direction.SOUTH:
            return (
                self.x_coord - ALIGNMENT_OFFSET_CELLS,
                self.y_coord - DEPTH_CLEARANCE_CELLS,
                Direction.NORTH,
            )
        elif self.image_side is Direction.NORTH:
            return (
                self.x_coord - ALIGNMENT_OFFSET_CELLS,
                self.y_coord + OBSTACLE_FOOTPRINT_CELLS + DEPTH_CLEARANCE_CELLS,
                Direction.SOUTH,
            )
        elif self.image_side is Direction.WEST:
            return (
                self.x_coord - DEPTH_CLEARANCE_CELLS,
                self.y_coord - ALIGNMENT_OFFSET_CELLS,
                Direction.EAST,
            )
        else:
            return (
                self.x_coord + OBSTACLE_FOOTPRINT_CELLS + DEPTH_CLEARANCE_CELLS,
                self.y_coord - ALIGNMENT_OFFSET_CELLS,
                Direction.WEST,
            )
            
    def inflated_bounds(self) -> tuple[float, float, float, float]:
        margin_cells = ROBOT_FOOTPRINT_CELLS / 2
        x_min = (self.x_coord - margin_cells) * GRID_LENGTH_CM
        y_min = (self.y_coord - margin_cells) * GRID_LENGTH_CM
        x_max = (self.x_coord + OBSTACLE_FOOTPRINT_CELLS + margin_cells) * GRID_LENGTH_CM
        y_max = (self.y_coord + OBSTACLE_FOOTPRINT_CELLS + margin_cells) * GRID_LENGTH_CM
        return (x_min, y_min, x_max, y_max)


@dataclass(frozen=True)
class Robot:
    x_coord: int
    y_coord: int
    facing: Direction

    def position(self) -> tuple[int, int, Direction]:
        return (self.x_coord, self.y_coord, self.facing)


@dataclass(frozen=True)
class Pose:
    x_cm: float
    y_cm: float
    theta_rad: float


@dataclass(frozen=True)
class Arena:
    obstacles = tuple[Obstacle, ...]

    def in_bounds(self, x_coord: int, y_coord: int) -> bool:
        return 0 <= x_coord < NUM_GRIDS and 0 <= y_coord < NUM_GRIDS

    def blocked_cells(self) -> set[tuple[int, int]]:
        cells = set()
        for obs in self.obstacles:
            for dx in range(OBSTACLE_FOOTPRINT_CELLS):
                for dy in range(OBSTACLE_FOOTPRINT_CELLS):
                    cells.add((obs.x_coord + dx, obs.y_coord + dy))
        return cells

    def is_free_cell(self, x_coord: int, y_coord: int) -> bool:
        return (
            self.in_bounds(x_coord, y_coord)
            and (x_coord, y_coord) not in self.blocked_cells()
        )
