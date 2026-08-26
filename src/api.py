"""
HTTP API exposing the Hamiltonian-path planner so another device (the RPi)
can call it over the network, instead of needing to import this code
directly. Run from inside src/:

    python api.py

Then from the RPi (or anywhere on the same network):

    POST http://<this-machine's-ip>:5000/plan
    {
      "robot": {"x_coord": 1, "y_coord": 1, "facing": "NORTH"},
      "obstacles": [{"id": "1", "x_coord": 5, "y_coord": 15, "image_side": "SOUTH"}],
      "radius": 25,              # optional, defaults to TURNING_RADIUS_CM
      "algorithm": "exhaustive_search"   # optional, one of the ALGORITHMS keys below
    }
"""

import time

from flask import Flask, jsonify, request

from algorithms.dubins import TURNING_RADIUS_CM, dubins_path, path_commands
from algorithms.hamiltonian import (
    exhaustive_search,
    nearest_neighbour,
    pairwise_swap,
    path_length,
)
from algorithms.graph import Graph
from model import parse_scenario

app = Flask(__name__)

ALGORITHMS = {
    "nearest_neighbour": nearest_neighbour,
    "pairwise_swap": pairwise_swap,
    "exhaustive_search": exhaustive_search,
}
DEFAULT_ALGORITHM = "exhaustive_search"


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/plan")
def plan():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="request body must be JSON"), 400

    try:
        robot, obstacles = parse_scenario(data)
    except (KeyError, TypeError) as exc:
        return jsonify(error=f"invalid robot/obstacles: {exc}"), 400

    if not obstacles:
        return jsonify(error="obstacles list must not be empty"), 400

    algorithm_name = data.get("algorithm", DEFAULT_ALGORITHM)
    algorithm = ALGORITHMS.get(algorithm_name)
    if algorithm is None:
        return (
            jsonify(
                error=f"unknown algorithm '{algorithm_name}', expected one of {list(ALGORITHMS)}"
            ),
            400,
        )

    radius = data.get("radius", TURNING_RADIUS_CM)

    graph = Graph.build(robot, obstacles)

    t0 = time.perf_counter()
    order = algorithm(graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    robots = {node.id: node.robot for node in graph.nodes}
    waypoints = [
        {
            "id": node_id,
            "x_cm": robots[node_id].x_cm,
            "y_cm": robots[node_id].y_cm,
            "theta_rad": robots[node_id].theta_rad,
        }
        for node_id in order
    ]

    legs = []
    for from_id, to_id in zip(order, order[1:]):
        path = dubins_path(robots[from_id], robots[to_id], radius)
        legs.append({"to": to_id, "commands": path_commands(path)})

    return jsonify(
        algorithm=algorithm_name,
        order=order,
        waypoints=waypoints,
        legs=legs,
        length_cm=path_length(graph, order),
        time_ms=elapsed_ms,
    )


if __name__ == "__main__":
    # host="0.0.0.0" so devices other than this machine (the RPi) can reach it -
    # the default 127.0.0.1 only accepts connections from this same machine.
    app.run(host="0.0.0.0", port=5000)
