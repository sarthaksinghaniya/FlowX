from __future__ import annotations

from typing import Dict


class DensityCalculator:
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, capacity: float = 30.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity

    def compute(self, lane_counts: Dict[str, int]) -> Dict[str, float]:
        densities: Dict[str, float] = {}
        for lane_name, vehicle_count in lane_counts.items():
            queue_estimate = vehicle_count * 0.5
            density = (self.alpha * vehicle_count + self.beta * queue_estimate) / self.capacity
            densities[lane_name] = density
        return densities
