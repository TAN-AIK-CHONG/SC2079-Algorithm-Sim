import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from collision import footprint_in_collision
from model import (
    NUM_GRIDS,
    Direction,
    Obstacle,
    Robot,
)
from planner import PlanningError, plan_mission

OUTPUT_DIR = ROOT_DIR / "testing" / "generated_maps"
OBSTACLE_COUNTS = (4, 5, 6, 7, 8)
MAPS_PER_COUNT = 3

ROBOT_START_CELL = (0, 0, Direction.NORTH)
ROBOT_START = Robot.from_grid(*ROBOT_START_CELL)
START_ZONE_CELLS = 4

# Accept a random layout once at least this fraction of its obstacles are
# reachable, rather than requiring every single one. Demanding 100% makes
# _generate_map()'s retry loop take a very long time (most random layouts
# have at least one obstacle hybrid_astar can't reach) - see
# plan_mission()'s skip-and-continue behaviour, which this counts against.
MIN_REACHABLE_FRACTION = 0.5


def _meets_reachability_bar(robot: Robot, obstacles: list[Obstacle]) -> bool:
    # plan_mission() itself, not a separate reimplementation (see
    # testing/pathing.py's history) - a map this accepts is only actually
    # useful for testing if the real production planner can solve it too.
    try:
        mission = plan_mission(robot, obstacles)
    except PlanningError:
        return False
    return len(mission.legs) >= len(obstacles) * MIN_REACHABLE_FRACTION


def _viewing_pose_ok(obstacle: Obstacle, obstacles: list[Obstacle]) -> bool:
    return not footprint_in_collision(
        obstacle.cm_viewing_position(),
        [other.footprint_corners_cm() for other in obstacles],
    )


def _viewing_poses_ok(obstacles: list[Obstacle]) -> bool:
    return all(_viewing_pose_ok(obstacle, obstacles) for obstacle in obstacles)


def _generate_random_obstacle(obstacle_id: int, rng: random.Random) -> Obstacle:
    while True:
        x_coord = rng.randint(0, NUM_GRIDS - 1)
        y_coord = rng.randint(0, NUM_GRIDS - 1)
        if x_coord <= START_ZONE_CELLS and y_coord <= START_ZONE_CELLS:
            continue
        image_side = rng.choice(list(Direction))
        return Obstacle(obstacle_id, x_coord, y_coord, image_side)


def _generate_obstacles(num_obstacles: int, rng: random.Random) -> list[Obstacle]:
    final_obstacles: list[Obstacle] = []
    occupied_cells: set[tuple[int, int]] = set()

    while len(final_obstacles) != num_obstacles:
        potential_obstacle = _generate_random_obstacle(len(final_obstacles), rng)
        cell = (potential_obstacle.x_coord, potential_obstacle.y_coord)
        if cell not in occupied_cells:
            occupied_cells.add(cell)
            final_obstacles.append(potential_obstacle)

    return final_obstacles


def _generate_map(
    robot: Robot, num_obstacles: int, rng: random.Random
) -> list[Obstacle]:
    obstacles = _generate_obstacles(num_obstacles, rng)
    while not (_viewing_poses_ok(obstacles) and _meets_reachability_bar(robot, obstacles)):
        obstacles = _generate_obstacles(num_obstacles, rng)

    return obstacles


def _map_to_dict(
    robot_cell: tuple[int, int, Direction], obstacles: list[Obstacle]
) -> dict:
    x_coord, y_coord, facing = robot_cell
    return {
        "robot": {
            "x_coord": x_coord,
            "y_coord": y_coord,
            "facing": facing.name,
        },
        "obstacles": [
            {
                "id": obs.id,
                "x_coord": obs.x_coord,
                "y_coord": obs.y_coord,
                "image_side": obs.image_side.name,
            }
            for obs in obstacles
        ],
    }


def main():
    for num_obstacles in OBSTACLE_COUNTS:
        out_dir = OUTPUT_DIR / f"{num_obstacles}_obstacles"
        out_dir.mkdir(parents=True, exist_ok=True)

        for map_index in range(1, MAPS_PER_COUNT + 1):
            seed = num_obstacles * 10_000 + map_index
            rng = random.Random(seed)
            obstacles = _generate_map(ROBOT_START, num_obstacles, rng)

            out_path = out_dir / f"map_{map_index:02d}.json"
            with out_path.open("w", encoding="utf-8") as file:
                json.dump(_map_to_dict(ROBOT_START_CELL, obstacles), file, indent=4)
                file.write("\n")

            print(f"wrote {out_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
