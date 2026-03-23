from __future__ import annotations

from typing import Dict


FIXED_CYCLE_TIME = 120
PROCESSING_FACTOR = 0.25


def _compute_metrics(
    green_times: Dict[str, int],
    lane_counts: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    wait_time: Dict[str, float] = {}
    queue_length: Dict[str, float] = {}
    throughput: Dict[str, float] = {}

    for lane_name, vehicles in lane_counts.items():
        green_time = max(1, green_times.get(lane_name, 1))
        processed = min(float(vehicles), green_time * PROCESSING_FACTOR)
        wait_time[lane_name] = round(float(vehicles) / green_time, 2)
        queue_length[lane_name] = round(max(0.0, float(vehicles) - processed), 2)
        throughput[lane_name] = round(processed, 2)

    summary = {
        "wait_time": round(sum(wait_time.values()), 2),
        "queue_length": round(sum(queue_length.values()), 2),
        "throughput": round(sum(throughput.values()), 2),
    }

    return {
        "per_lane": {
            "wait_time": wait_time,
            "queue_length": queue_length,
            "throughput": throughput,
        },
        "summary": summary,
    }


def simulate_static(lane_counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    lane_total = max(1, len(lane_counts))
    equal_green = max(1, FIXED_CYCLE_TIME // lane_total)
    static_green_times = {lane_name: equal_green for lane_name in lane_counts}
    return _compute_metrics(static_green_times, lane_counts)


def simulate_ai(
    green_times: Dict[str, int],
    lane_counts: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    ai_green_times = {
        lane_name: max(1, int(green_times.get(lane_name, 1)))
        for lane_name in lane_counts
    }
    return _compute_metrics(ai_green_times, lane_counts)
