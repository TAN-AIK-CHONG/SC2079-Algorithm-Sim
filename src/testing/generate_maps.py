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
from visualize_map import draw, plan

OUTPUT_DIR = ROOT_DIR / "testing" / "generated_maps"
PNG_OUTPUT_DIR = ROOT_DIR / "testing" / "generated_maps_png"
FAILURES_PATH = OUTPUT_DIR / "failures.txt"
OBSTACLE_COUNTS = (7, 8)
MAPS_PER_COUNT = 50

ROBOT_START_CELL = (0, 0, Direction.NORTH)
ROBOT_START = Robot.from_grid(*ROBOT_START_CELL)
START_ZONE_CELLS = 4


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


def _viewing_poses_ok(obstacles: list[Obstacle]) -> bool:
    footprints = [
        obstacle.inflated_footprint_corners_cm() for obstacle in obstacles
    ]
    return all(
        not footprint_in_collision(obstacle.cm_viewing_position(), footprints)
        for obstacle in obstacles
    )


def _generate_map(num_obstacles: int, rng: random.Random) -> list[Obstacle]:
    obstacles = _generate_obstacles(num_obstacles, rng)
    while not _viewing_poses_ok(obstacles):
        obstacles = _generate_obstacles(num_obstacles, rng)
    return obstacles


def _plan_summary(plan_result, num_obstacles: int) -> tuple[bool, str]:
    """(every obstacle reached?, one-line summary) for a visualize_map.plan()
    result.

    A skipped obstacle counts as a failure here just as a total planning
    failure does - in the real run both mean an obstacle never got
    photographed, so both belong in the failure log.
    """
    _, final_path, length_cm, completed_legs, skipped_ids = plan_result
    if not final_path:
        return False, f"0/{num_obstacles} reachable - nothing could be planned at all"

    summary = f"{completed_legs}/{num_obstacles} reachable, {length_cm:.0f}cm"
    if skipped_ids:
        summary += f" (skipped {', '.join(str(i) for i in skipped_ids)})"
    return not skipped_ids, summary


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    # Written a line at a time and flushed, not collected and dumped at the
    # end: this run takes long enough to leave in the background, so the log
    # has to be readable while it is still going and has to survive the run
    # being killed part-way.
    with FAILURES_PATH.open("w", encoding="utf-8") as failure_log:
        for num_obstacles in OBSTACLE_COUNTS:
            out_dir = OUTPUT_DIR / f"{num_obstacles}_obstacles"
            out_dir.mkdir(parents=True, exist_ok=True)
            png_dir = PNG_OUTPUT_DIR / f"{num_obstacles}_obstacles"
            png_dir.mkdir(parents=True, exist_ok=True)

            for map_index in range(1, MAPS_PER_COUNT + 1):
                seed = num_obstacles * 10_000 + map_index
                obstacles = _generate_map(num_obstacles, random.Random(seed))

                name = f"map_{map_index:03d}"
                out_path = out_dir / f"{name}.json"
                with out_path.open("w", encoding="utf-8") as file:
                    json.dump(_map_to_dict(ROBOT_START_CELL, obstacles), file, indent=4)
                    file.write("\n")

                # One plan per map, shared by the failure log and the PNG -
                # planning dominates this script's runtime, so the picture is
                # drawn from the same result rather than re-planning.
                plan_result = plan(ROBOT_START, obstacles)
                png_path = png_dir / f"{name}.png"
                draw(name, ROBOT_START, obstacles, plan_result, png_path)

                solved, summary = _plan_summary(plan_result, num_obstacles)
                line = f"{out_path.relative_to(ROOT_DIR)}  {summary}"
                print(f"wrote {line} + {png_path.relative_to(ROOT_DIR)}", flush=True)

                if not solved:
                    failures += 1
                    failure_log.write(line + "\n")
                    failure_log.flush()

    print(
        f"\n{failures}/{len(OBSTACLE_COUNTS) * MAPS_PER_COUNT} maps not fully solved"
        f" -> {FAILURES_PATH.relative_to(ROOT_DIR)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
