from __future__ import annotations

from typing import Dict


class SignalController:
    def __init__(self, cycle_time: int = 120, minimum_green: int = 10) -> None:
        self.cycle_time = cycle_time
        self.minimum_green = minimum_green

    def allocate_green_times(self, lane_densities: Dict[str, float]) -> Dict[str, int]:
        if not lane_densities:
            return {}

        total_density = sum(lane_densities.values())
        if total_density <= 0:
            return {lane_name: self.minimum_green for lane_name in lane_densities}

        provisional = {
            lane_name: max(
                self.minimum_green,
                int(round((density / total_density) * self.cycle_time)),
            )
            for lane_name, density in lane_densities.items()
        }

        total_allocated = sum(provisional.values())
        if total_allocated == self.cycle_time:
            return provisional

        adjusted = provisional.copy()
        if total_allocated < self.cycle_time:
            self._distribute_remaining_time(adjusted, lane_densities, self.cycle_time - total_allocated)
            return adjusted

        self._reduce_excess_time(adjusted, lane_densities, total_allocated - self.cycle_time)
        return adjusted

    def _distribute_remaining_time(
        self,
        allocations: Dict[str, int],
        lane_densities: Dict[str, float],
        remaining_seconds: int,
    ) -> None:
        sorted_lanes = sorted(lane_densities, key=lane_densities.get, reverse=True)
        index = 0
        while remaining_seconds > 0 and sorted_lanes:
            lane_name = sorted_lanes[index % len(sorted_lanes)]
            allocations[lane_name] += 1
            remaining_seconds -= 1
            index += 1

    def _reduce_excess_time(
        self,
        allocations: Dict[str, int],
        lane_densities: Dict[str, float],
        excess_seconds: int,
    ) -> None:
        sorted_lanes = sorted(lane_densities, key=lane_densities.get)
        while excess_seconds > 0 and sorted_lanes:
            progress_made = False
            for lane_name in sorted_lanes:
                if allocations[lane_name] > self.minimum_green and excess_seconds > 0:
                    allocations[lane_name] -= 1
                    excess_seconds -= 1
                    progress_made = True
            if not progress_made:
                break
