import json
from pathlib import Path

from graph import Graph
from algorithms.hamiltonian import run_all
from model import Direction, Obstacle, Robot

TEST_MAPS_DIR = Path(__file__).parent.parent / "test_maps"


def load_map(path: Path) -> tuple[Robot, list[Obstacle]]:
    with open(path) as f:
        data = json.load(f)

    robot = Robot(
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


def main():
    map_paths = sorted(TEST_MAPS_DIR.glob("*.json"))

    for map_path in map_paths:
        robot, obstacles = load_map(map_path)
        graph = Graph.build(robot, obstacles, radius=25)
        results = run_all(graph)

        print(f"Map: {map_path.name} ({len(obstacles)} obstacles)\n")
        for name, r in results.items():
            print(
                f"{name:>18}: path={r['path']}  "
                f"length={r['length']:.2f}cm  "
                f"time={r['time']*1000:.3f}ms  "
                f"accuracy={r['accuracy']:.2%}"
            )
        print("---")


if __name__ == "__main__":
    main()