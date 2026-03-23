from __future__ import annotations

from collections import deque
from typing import Dict, List, Mapping

from src.graph import ROAD_NETWORK


def compute_path(
    start: str,
    end: str,
    graph: Mapping[str, List[str]] | None = None,
) -> List[str]:
    network = dict(graph or ROAD_NETWORK)
    if start == end:
        return [start]
    if start not in network or end not in network:
        return []

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        for neighbor in network.get(node, []):
            if neighbor in visited:
                continue
            next_path = [*path, neighbor]
            if neighbor == end:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))

    return []


def generate_corridor(path: List[str]) -> Dict[str, str]:
    if not path:
        return {}

    corridor_states: Dict[str, str] = {}
    for index, node in enumerate(path):
        if index == 0:
            corridor_states[node] = "GREEN"
        elif index == 1:
            corridor_states[node] = "PREPARE"
        else:
            corridor_states[node] = "WAIT"
    return corridor_states
