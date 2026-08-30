"""Render a map JSON (arena grid, obstacles, planned path) to a PNG image.

For a single map:
    python visualize_map.py generated_maps/4_obstacles/map_01.json

For every map under a directory (recurses into subfolders):
    python visualize_map.py generated_maps --out generated_maps_png
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from algorithms.hamiltonian import run_all
from algorithms.graph import Graph
from model import (
    ARENA_LENGTH_CM,
    GRID_LENGTH_CM,
    NUM_GRIDS,
    OBSTACLE_FOOTPRINT_CELLS,
    Direction,
    parse_scenario,
)
from testing.pathing import calculate_final_path

# Points in the direction the obstacle's image faces, drawn as an arrow off the obstacle cell.
IMAGE_SIDE_OFFSET = {
    Direction.NORTH: (0, 1),
    Direction.SOUTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}


def load_map(path: Path):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return parse_scenario(data)


def plan(robot, obstacles):
    graph = Graph.build(robot, obstacles)
    order = run_all(graph)["exhaustive_search"]["path"]
    final_path, length_cm, completed_legs, skipped_legs = calculate_final_path(
        graph, obstacles, order
    )
    return order, final_path, length_cm, completed_legs, skipped_legs


def render(map_path: Path, out_path: Path):
    robot, obstacles = load_map(map_path)
    order, final_path, length_cm, completed_legs, skipped_legs = plan(robot, obstacles)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, ARENA_LENGTH_CM)
    ax.set_ylim(0, ARENA_LENGTH_CM)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, ARENA_LENGTH_CM + 1, GRID_LENGTH_CM), minor=False)
    ax.set_yticks(range(0, ARENA_LENGTH_CM + 1, GRID_LENGTH_CM), minor=False)
    ax.set_xticklabels(range(0, NUM_GRIDS + 1))
    ax.set_yticklabels(range(0, NUM_GRIDS + 1))
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)

    obstacle_size_cm = OBSTACLE_FOOTPRINT_CELLS * GRID_LENGTH_CM
    for obs in obstacles:
        x_cm, y_cm = obs.x_coord * GRID_LENGTH_CM, obs.y_coord * GRID_LENGTH_CM
        ax.add_patch(
            Rectangle(
                (x_cm, y_cm),
                obstacle_size_cm,
                obstacle_size_cm,
                facecolor="#444444",
                edgecolor="black",
                zorder=3,
            )
        )
        ax.text(
            x_cm + obstacle_size_cm / 2,
            y_cm + obstacle_size_cm / 2,
            obs.id,
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
        )
        # Red arrow off the face carrying the image, to show image_side.
        dx, dy = IMAGE_SIDE_OFFSET[obs.image_side]
        ax.annotate(
            "",
            xy=(
                x_cm + obstacle_size_cm / 2 + dx * 8,
                y_cm + obstacle_size_cm / 2 + dy * 8,
            ),
            xytext=(x_cm + obstacle_size_cm / 2, y_cm + obstacle_size_cm / 2),
            arrowprops=dict(arrowstyle="->", color="red", linewidth=2),
            zorder=5,
        )

    if final_path:
        xs = [robot.x_cm for robot in final_path]
        ys = [robot.y_cm for robot in final_path]
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.5, zorder=2, label="planned path")

    # Viewing pose markers, in visit order.
    node_robots = {node.id: node.viewing_pose for node in Graph.build(robot, obstacles).nodes}
    for visit_index, node_id in enumerate(order):
        node_robot = node_robots[node_id]
        label = "S" if node_id == "S" else node_id
        color = "green" if node_id == "S" else "orange"
        ax.plot(
            node_robot.x_cm,
            node_robot.y_cm,
            marker="o",
            color=color,
            markersize=6,
            zorder=6,
        )
        ax.annotate(
            f"{visit_index}:{label}",
            (node_robot.x_cm, node_robot.y_cm),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            zorder=6,
        )

    status = (
        "complete"
        if not skipped_legs
        else f"skipped {len(skipped_legs)}: " + ", ".join(f"{a}->{b}" for a, b in skipped_legs)
    )
    ax.set_title(
        f"{map_path.stem}  |  {len(obstacles)} obstacles  |  "
        f"order: {' -> '.join(str(node_id) for node_id in order)}\n"
        f"length={length_cm:.1f}cm  legs={completed_legs}/{len(order) - 1}  {status}",
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="a map JSON file, or a directory to recurse into"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: alongside input)",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        map_paths = sorted(args.input.rglob("*.json"))
        if not map_paths:
            raise SystemExit(f"no *.json maps found under {args.input}")
        out_root = args.out or args.input
        for map_path in map_paths:
            rel = map_path.relative_to(args.input)
            out_path = (out_root / rel).with_suffix(".png")
            render(map_path, out_path)
            print(f"wrote {out_path}")
    else:
        out_path = (
            args.out / args.input.with_suffix(".png").name
            if args.out
            else args.input.with_suffix(".png")
        )
        render(args.input, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
