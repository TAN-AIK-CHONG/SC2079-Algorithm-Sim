from dataclasses import dataclass
from enum import Enum

NUM_GRIDS = (40, 40)
GRID_LENGTH_CM = 5

ROBOT_TRUE_SIZE_CM = (20, 21)
ROBOT_FOOTPRINT_CM = (30, 30)
OBSTACLE_SIZE_CM = (10, 10)
OBSTACLE_SIZE = (2, 2)


class Direction(Enum):
    NORTH = (0, 1)
    SOUTH = (1, 0)
    EAST = (0, -1)
    WEST = (-1, 0)


@dataclass(frozen=True)
class Obstacle:
    id: int
    x: int
    y: int
    image_side: Direction


@dataclass(frozen=True)
class State:
    x: int
    y: int
    facing: Direction

    def position(self) -> tuple[int, int]:
        return (self.x, self.y)
