"""
Plans a full mission: the best order to visit obstacles in, and the drive
commands for each leg between them. Called directly, in-process, by the
RPi's own code (see rpi/main.py) - there is no HTTP/network step anymore,
the algorithm lives inside the RPi codebase.
"""

import time

from algorithms.dubins import TURNING_RADIUS_CM, dubins_path, path_commands
from algorithms.graph import Graph
from algorithms.hamiltonian import exhaustive_search, nearest_neighbour, pairwise_swap, path_length
from model import Obstacle, Robot

ALGORITHMS = {
    "nearest_neighbour": nearest_neighbour,
    "pairwise_swap": pairwise_swap,
    "exhaustive_search": exhaustive_search,
}
DEFAULT_ALGORITHM = "exhaustive_search"  # obstacle count is small enough that this is cheap and optimal


def plan_mission(
    robot: Robot,
    obstacles: list[Obstacle],
    algorithm: str = DEFAULT_ALGORITHM,
    radius: float = TURNING_RADIUS_CM,
) -> dict:
    if not obstacles:
        raise ValueError("obstacles list must not be empty")

    algorithm_fn = ALGORITHMS.get(algorithm)
    if algorithm_fn is None:
        raise ValueError(f"unknown algorithm '{algorithm}', expected one of {list(ALGORITHMS)}")

    # Graph.build() always weighs edges at the default TURNING_RADIUS_CM (it
    # no longer takes a radius argument); `radius` below only controls the
    # actual leg commands generated for driving.
    graph = Graph.build(robot, obstacles)

    t0 = time.perf_counter()
    order = algorithm_fn(graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    robots = {node.id: node.robot for node in graph.nodes}
    waypoints = [
        {"id": node_id, "x_cm": robots[node_id].x_cm, "y_cm": robots[node_id].y_cm, "theta_rad": robots[node_id].theta_rad}
        for node_id in order
    ]

    legs = []
    for from_id, to_id in zip(order, order[1:]):
        path = dubins_path(robots[from_id], robots[to_id], radius)
        legs.append({"to": to_id, "commands": path_commands(path)})

    return {
        "algorithm": algorithm,
        "order": order,
        "waypoints": waypoints,
        "legs": legs,
        "length_cm": path_length(graph, order),
        "time_ms": elapsed_ms,
    }
