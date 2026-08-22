import itertools

from graph import Graph


def path_length(graph: Graph, path: list[str]) -> float:
    return sum(
        graph.edge_weight(path[i], path[i + 1]) for i in range(len(path) - 1)
    )


def nearest_neighbour(graph: Graph) -> list[str]:
    unvisited = {node.id for node in graph.nodes if node.id != "S"}
    path = ["S"]
    current = "S"

    while unvisited:
        nearest = min(unvisited, key=lambda w: graph.edge_weight(current, w))
        path.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return path


def pairwise_swap(graph: Graph) -> list[str]:
    best = nearest_neighbour(graph)
    best_length = path_length(graph, best)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            candidate = best.copy()
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
            candidate_length = path_length(graph, candidate)
            if candidate_length < best_length:
                best, best_length = candidate, candidate_length
                improved = True

    return best


def exhaustive_search(graph: Graph) -> list[str]:
    obstacle_ids = [node.id for node in graph.nodes if node.id != "S"]

    best = None
    best_length = float("inf")
    for perm in itertools.permutations(obstacle_ids):
        candidate = ["S", *perm]
        candidate_length = path_length(graph, candidate)
        if candidate_length < best_length:
            best, best_length = candidate, candidate_length

    return best