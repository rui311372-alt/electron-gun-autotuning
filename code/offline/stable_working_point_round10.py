#!/usr/bin/env python3
"""Offline stable-working-point identification for the electron-gun study.

The routine intentionally does not alter the online electron-gun controller.
It identifies the terminal low-CV branch from completed runs and then selects
the measured Uc with the lowest candidate-level score inside that branch.

Main steps
----------
1. Sort valid candidates by measured/set Uc.
2. Compute z_i = log(CV_i) from all three valid shots.
3. Segment z_i with dynamic programming under a Laplace likelihood; choose the
   number of segments by BIC.
4. Divide the segment medians into low- and high-CV states with deterministic,
   weighted one-dimensional two-means clustering.
5. Starting at the highest-Uc end, collect the contiguous low-CV segments.
6. In that terminal stable branch, choose the measured candidate with minimum
   score (ties are resolved toward the lower Uc).

The default minimum segment length is four candidates. It prevents isolated
one-to-three-point fluctuations from becoming their own regime. The companion
sensitivity output repeats the analysis with lengths 3, 4, and 5, two loss
models, and three candidate-score aggregation rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


EPS = 1e-6
AGGREGATIONS = ("closest_pair", "median3", "mean3")
COST_MODELS = ("laplace", "student_t4", "gaussian")
PENALTY_MODES = ("segment_parameters", "segment_plus_breakpoints")


@dataclass(frozen=True)
class Candidate:
    run: str
    iteration: int
    uc: float
    feedback_uc: float
    shots: tuple[float, float, float]
    cv: float

    def score(self, aggregation: str) -> float:
        if aggregation == "closest_pair":
            pairs = ((0, 1), (0, 2), (1, 2))
            p, q = min(
                pairs,
                key=lambda pair: (
                    relative_difference(self.shots[pair[0]], self.shots[pair[1]]),
                    pair,
                ),
            )
            return (self.shots[p] + self.shots[q]) / 2.0
        if aggregation == "median3":
            return float(statistics.median(self.shots))
        if aggregation == "mean3":
            return float(statistics.mean(self.shots))
        raise ValueError(f"Unknown aggregation: {aggregation}")


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    uc_min: float
    uc_max: float
    count: int
    median_cv: float
    median_log_cv: float
    state: str = ""


@dataclass(frozen=True)
class WorkingPointResult:
    run: str
    aggregation: str
    cost_model: str
    penalty_mode: str
    min_segment_size: int
    candidate_count: int
    segment_count: int
    transition_lower_uc: float
    transition_upper_uc: float
    stable_start_uc: float
    optimum_uc: float
    optimum_score: float
    stable_candidate_count: int
    pre_transition_median_cv: float
    stable_branch_median_cv: float
    low_state_center_log_cv: float
    high_state_center_log_cv: float
    segments: tuple[Segment, ...]


def finite_number(value: object, field: str) -> float:
    if value is None:
        raise ValueError(f"Missing {field}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Blank {field}")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return number


def relative_difference(a: float, b: float) -> float:
    denominator = min(abs(a), abs(b))
    if denominator == 0:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / denominator


def sample_cv(values: Sequence[float]) -> float:
    if len(values) < 3:
        raise ValueError("Three valid shots are required for post-run CV analysis")
    mean = statistics.mean(values)
    if mean == 0:
        raise ValueError("Cannot compute CV when the three-shot mean is zero")
    return statistics.stdev(values) / abs(mean) * 100.0


def load_candidates(path: Path) -> tuple[dict[str, list[Candidate]], list[str]]:
    runs: dict[str, list[Candidate]] = {}
    excluded: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"run", "iteration", "uc_set", "uc_feedback", "shot1", "shot2", "shot3"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                run = str(row.get("run", "")).strip()
                if not run:
                    raise ValueError("Blank run")
                shots = tuple(finite_number(row.get(f"shot{i}"), f"shot{i}") for i in range(1, 4))
                candidate = Candidate(
                    run=run,
                    iteration=int(finite_number(row.get("iteration"), "iteration")),
                    uc=finite_number(row.get("uc_set"), "uc_set"),
                    feedback_uc=finite_number(row.get("uc_feedback"), "uc_feedback"),
                    shots=(shots[0], shots[1], shots[2]),
                    cv=sample_cv(shots),
                )
            except (TypeError, ValueError) as exc:
                excluded.append(f"row {row_number}: {exc}")
                continue
            runs.setdefault(candidate.run, []).append(candidate)
    for values in runs.values():
        values.sort(key=lambda item: (item.uc, item.iteration))
    return runs, excluded


def _segment_cost(values: Sequence[float], model: str) -> float:
    n = len(values)
    if n == 0:
        return float("inf")
    if model == "laplace":
        location = statistics.median(values)
        scale = max(statistics.mean(abs(value - location) for value in values), EPS)
        # -2 log likelihood for a Laplace(location, scale) segment.
        return 2.0 * n * (math.log(2.0 * scale) + 1.0)
    if model == "gaussian":
        location = statistics.mean(values)
        variance = max(statistics.mean((value - location) ** 2 for value in values), EPS)
        # -2 log likelihood for a Gaussian segment.
        return n * (math.log(2.0 * math.pi * variance) + 1.0)
    if model == "student_t4":
        # Fixed-nu Student-t maximum likelihood by the standard EM updates.
        # nu is fixed at four, so each segment still estimates two continuous
        # parameters (location and scale), matching the main BIC parameter count.
        nu = 4.0
        location = float(statistics.median(values))
        mad = statistics.median(abs(value - location) for value in values)
        scale = max(1.4826 * mad, statistics.pstdev(values), EPS)
        for _ in range(200):
            old_location, old_scale = location, scale
            weights = [
                (nu + 1.0) / (nu + ((value - location) / scale) ** 2)
                for value in values
            ]
            weight_sum = sum(weights)
            location = sum(weight * value for weight, value in zip(weights, values)) / weight_sum
            variance = sum(
                weight * (value - location) ** 2
                for weight, value in zip(weights, values)
            ) / n
            scale = max(math.sqrt(variance), EPS)
            if abs(location - old_location) <= 1e-10 and abs(scale - old_scale) <= 1e-10:
                break
        constant = (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * math.log(nu * math.pi)
            - math.log(scale)
        )
        log_likelihood = sum(
            constant
            - ((nu + 1.0) / 2.0)
            * math.log1p(((value - location) / scale) ** 2 / nu)
            for value in values
        )
        return -2.0 * log_likelihood
    raise ValueError(f"Unknown cost model: {model}")


def bic_segmentation(
    log_cv: Sequence[float],
    *,
    min_segment_size: int = 4,
    cost_model: str = "laplace",
    penalty_mode: str = "segment_parameters",
) -> list[tuple[int, int]]:
    """Return BIC-selected contiguous segments as half-open index intervals."""
    n = len(log_cv)
    if min_segment_size < 2:
        raise ValueError("min_segment_size must be at least 2")
    if penalty_mode not in PENALTY_MODES:
        raise ValueError(f"penalty_mode must be one of {PENALTY_MODES}")
    if n < 2 * min_segment_size:
        raise ValueError(
            f"At least {2 * min_segment_size} candidates are required to identify two regimes"
        )
    max_segments = n // min_segment_size
    infinity = float("inf")

    costs = [[infinity] * (n + 1) for _ in range(n)]
    for start in range(n):
        for stop in range(start + min_segment_size, n + 1):
            costs[start][stop] = _segment_cost(log_cv[start:stop], cost_model)

    dp = [[infinity] * (n + 1) for _ in range(max_segments + 1)]
    previous: list[list[int | None]] = [[None] * (n + 1) for _ in range(max_segments + 1)]
    dp[0][0] = 0.0
    for segment_count in range(1, max_segments + 1):
        first_stop = segment_count * min_segment_size
        for stop in range(first_stop, n + 1):
            first_start = (segment_count - 1) * min_segment_size
            last_start = stop - min_segment_size
            for start in range(first_start, last_start + 1):
                value = dp[segment_count - 1][start] + costs[start][stop]
                if value < dp[segment_count][stop]:
                    dp[segment_count][stop] = value
                    previous[segment_count][stop] = start

    candidates: list[tuple[float, int]] = []
    for segment_count in range(1, max_segments + 1):
        if not math.isfinite(dp[segment_count][n]):
            continue
        if penalty_mode == "segment_parameters":
            # Main analysis: two continuous parameters per segment. Discrete
            # breakpoint locations are searched by DP and not counted here.
            penalty_parameters = 2 * segment_count
        else:
            # Sensitivity analysis also counts K-1 discrete breakpoints.
            penalty_parameters = 3 * segment_count - 1
        bic = dp[segment_count][n] + penalty_parameters * math.log(n)
        candidates.append((bic, segment_count))
    if not candidates:
        raise RuntimeError("No valid segmentation was found")
    _, chosen_count = min(candidates, key=lambda item: (item[0], item[1]))

    boundaries = [n]
    stop = n
    for segment_count in range(chosen_count, 0, -1):
        start = previous[segment_count][stop]
        if start is None:
            raise RuntimeError("Segmentation backtracking failed")
        boundaries.append(start)
        stop = start
    boundaries.reverse()
    return list(zip(boundaries[:-1], boundaries[1:]))


def _weighted_sse(values: Sequence[float], weights: Sequence[int]) -> tuple[float, float]:
    total_weight = sum(weights)
    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    sse = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights))
    return sse, mean


def classify_low_high_segments(segments: Sequence[Segment]) -> tuple[list[str], float, float]:
    """Classify segment medians into deterministic low/high CV states."""
    if len(segments) < 2:
        raise ValueError("At least two segments are required to identify low/high CV states")
    order = sorted(range(len(segments)), key=lambda idx: segments[idx].median_log_cv)
    sorted_values = [segments[idx].median_log_cv for idx in order]
    sorted_weights = [segments[idx].count for idx in order]
    best: tuple[float, int, float, float] | None = None
    for split in range(1, len(segments)):
        left_sse, left_mean = _weighted_sse(sorted_values[:split], sorted_weights[:split])
        right_sse, right_mean = _weighted_sse(sorted_values[split:], sorted_weights[split:])
        candidate = (left_sse + right_sse, split, left_mean, right_mean)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("Two-state clustering failed")
    _, split, low_center, high_center = best
    low_indices = set(order[:split])
    labels = ["low" if idx in low_indices else "high" for idx in range(len(segments))]
    return labels, low_center, high_center


def identify_working_point(
    candidates: Sequence[Candidate],
    *,
    aggregation: str = "closest_pair",
    min_segment_size: int = 4,
    cost_model: str = "laplace",
    penalty_mode: str = "segment_parameters",
    prefix_n: int | None = None,
) -> WorkingPointResult:
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
    if cost_model not in COST_MODELS:
        raise ValueError(f"cost_model must be one of {COST_MODELS}")
    if penalty_mode not in PENALTY_MODES:
        raise ValueError(f"penalty_mode must be one of {PENALTY_MODES}")
    selected = list(candidates)
    if prefix_n is not None:
        selected = sorted(selected, key=lambda item: item.iteration)[:prefix_n]
    selected.sort(key=lambda item: (item.uc, item.iteration))
    if len(selected) < 2 * min_segment_size:
        raise ValueError("Too few candidates for the requested segmentation")

    # CV enters as a percentage value (e.g. 1.01 rather than 0.0101).
    log_cv = [math.log(candidate.cv + EPS) for candidate in selected]
    intervals = bic_segmentation(
        log_cv,
        min_segment_size=min_segment_size,
        cost_model=cost_model,
        penalty_mode=penalty_mode,
    )
    raw_segments = [
        Segment(
            start=start,
            stop=stop,
            uc_min=selected[start].uc,
            uc_max=selected[stop - 1].uc,
            count=stop - start,
            median_cv=float(statistics.median(candidate.cv for candidate in selected[start:stop])),
            median_log_cv=float(statistics.median(log_cv[start:stop])),
        )
        for start, stop in intervals
    ]
    labels, low_center, high_center = classify_low_high_segments(raw_segments)
    segments = [
        Segment(**{**asdict(segment), "state": label})
        for segment, label in zip(raw_segments, labels)
    ]
    if segments[-1].state != "low":
        raise ValueError("The highest-Uc terminal segment is not a low-CV state")

    stable_segment_index = len(segments) - 1
    while stable_segment_index > 0 and segments[stable_segment_index - 1].state == "low":
        stable_segment_index -= 1
    if stable_segment_index == 0:
        raise ValueError("No preceding high-CV regime was identified")
    stable_start_index = segments[stable_segment_index].start
    transition_lower_index = stable_start_index - 1
    stable_candidates = selected[stable_start_index:]
    optimum = min(
        stable_candidates,
        key=lambda candidate: (candidate.score(aggregation), candidate.uc, candidate.iteration),
    )
    pre_candidates = selected[:stable_start_index]

    return WorkingPointResult(
        run=selected[0].run,
        aggregation=aggregation,
        cost_model=cost_model,
        penalty_mode=penalty_mode,
        min_segment_size=min_segment_size,
        candidate_count=len(selected),
        segment_count=len(segments),
        transition_lower_uc=selected[transition_lower_index].uc,
        transition_upper_uc=selected[stable_start_index].uc,
        stable_start_uc=selected[stable_start_index].uc,
        optimum_uc=optimum.uc,
        optimum_score=optimum.score(aggregation),
        stable_candidate_count=len(stable_candidates),
        pre_transition_median_cv=float(statistics.median(candidate.cv for candidate in pre_candidates)),
        stable_branch_median_cv=float(statistics.median(candidate.cv for candidate in stable_candidates)),
        low_state_center_log_cv=low_center,
        high_state_center_log_cv=high_center,
        segments=tuple(segments),
    )


def result_row(result: WorkingPointResult, *, prefix_n: int | None = None) -> dict[str, object]:
    return {
        "run": result.run,
        "prefix_n": "all" if prefix_n is None else prefix_n,
        "aggregation": result.aggregation,
        "cost_model": result.cost_model,
        "penalty_mode": result.penalty_mode,
        "min_segment_size": result.min_segment_size,
        "candidate_count": result.candidate_count,
        "segment_count": result.segment_count,
        "transition_lower_uc": result.transition_lower_uc,
        "transition_upper_uc": result.transition_upper_uc,
        "stable_start_uc": result.stable_start_uc,
        "optimum_uc": result.optimum_uc,
        "optimum_score": result.optimum_score,
        "stable_candidate_count": result.stable_candidate_count,
        "pre_transition_median_cv": result.pre_transition_median_cv,
        "stable_branch_median_cv": result.stable_branch_median_cv,
        "low_state_center_log_cv": result.low_state_center_log_cv,
        "high_state_center_log_cv": result.high_state_center_log_cv,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def sensitivity_row(
    scenario: str,
    run: str,
    *,
    aggregation: str = "closest_pair",
    cost_model: str = "laplace",
    min_segment_size: int = 4,
    penalty_mode: str = "segment_parameters",
    bin_width_v: object = "",
    excluded_uc_v: object = "",
    result: WorkingPointResult | None = None,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "run": run,
        "aggregation": aggregation,
        "cost_model": cost_model,
        "min_segment_size": min_segment_size,
        "penalty_mode": penalty_mode,
        "bin_width_v": bin_width_v,
        "bin_count": "",
        "excluded_uc_v": excluded_uc_v,
        "transition_lower_uc": "" if result is None else result.transition_lower_uc,
        "transition_upper_uc": "" if result is None else result.transition_upper_uc,
        "segment_count": "" if result is None else result.segment_count,
        "optimum_uc": "" if result is None else result.optimum_uc,
        "optimum_score": "" if result is None else result.optimum_score,
        "status": status,
    }


def identify_binned_working_point(
    candidates: Sequence[Candidate],
    *,
    bin_width_v: float = 5.0,
    aggregation: str = "closest_pair",
    min_segment_size: int = 4,
    cost_model: str = "laplace",
    penalty_mode: str = "segment_parameters",
) -> dict[str, object]:
    """Equalize adaptive sampling density by one median-CV observation per voltage bin."""
    if bin_width_v <= 0:
        raise ValueError("bin_width_v must be positive")
    selected = sorted(candidates, key=lambda item: (item.uc, item.iteration))
    grouped: dict[int, list[Candidate]] = {}
    for candidate in selected:
        key = math.floor(candidate.uc / bin_width_v)
        grouped.setdefault(key, []).append(candidate)
    bins = []
    for key, members in sorted(grouped.items()):
        bins.append(
            {
                "key": key,
                "lower": key * bin_width_v,
                "uc": float(statistics.median(item.uc for item in members)),
                "cv": float(statistics.median(item.cv for item in members)),
                "members": members,
            }
        )
    if len(bins) < 2 * min_segment_size:
        raise ValueError("Too few non-empty voltage bins for segmentation")
    log_cv = [math.log(float(item["cv"]) + EPS) for item in bins]
    intervals = bic_segmentation(
        log_cv,
        min_segment_size=min_segment_size,
        cost_model=cost_model,
        penalty_mode=penalty_mode,
    )
    raw_segments = [
        Segment(
            start=start,
            stop=stop,
            uc_min=float(bins[start]["uc"]),
            uc_max=float(bins[stop - 1]["uc"]),
            count=stop - start,
            median_cv=float(statistics.median(float(item["cv"]) for item in bins[start:stop])),
            median_log_cv=float(statistics.median(log_cv[start:stop])),
        )
        for start, stop in intervals
    ]
    labels, _, _ = classify_low_high_segments(raw_segments)
    segments = [
        Segment(**{**asdict(segment), "state": label})
        for segment, label in zip(raw_segments, labels)
    ]
    if segments[-1].state != "low":
        raise ValueError("The highest-Uc terminal segment is not a low-CV state")
    stable_segment_index = len(segments) - 1
    while stable_segment_index > 0 and segments[stable_segment_index - 1].state == "low":
        stable_segment_index -= 1
    if stable_segment_index == 0:
        raise ValueError("No preceding high-CV regime was identified")
    stable_bin_index = segments[stable_segment_index].start
    stable_lower_edge = float(bins[stable_bin_index]["lower"])
    stable_candidates = [item for item in selected if item.uc >= stable_lower_edge]
    optimum = min(
        stable_candidates,
        key=lambda item: (item.score(aggregation), item.uc, item.iteration),
    )
    return {
        "transition_lower_uc": float(bins[stable_bin_index - 1]["uc"]),
        "transition_upper_uc": float(bins[stable_bin_index]["uc"]),
        "segment_count": len(segments),
        "optimum_uc": optimum.uc,
        "optimum_score": optimum.score(aggregation),
        "bin_count": len(bins),
    }


def run_sensitivity(runs: dict[str, list[Candidate]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run, candidates in runs.items():
        # Candidate-level analysis: aggregation, minimum segment size, and loss model.
        for aggregation in AGGREGATIONS:
            for cost_model in COST_MODELS:
                for minimum in (3, 4, 5):
                    try:
                        result = identify_working_point(
                            candidates,
                            aggregation=aggregation,
                            min_segment_size=minimum,
                            cost_model=cost_model,
                        )
                    except ValueError as exc:
                        rows.append(
                            sensitivity_row(
                                "candidate_level",
                                run,
                                aggregation=aggregation,
                                cost_model=cost_model,
                                min_segment_size=minimum,
                                result=None,
                                status=str(exc),
                            )
                        )
                    else:
                        rows.append(
                            sensitivity_row(
                                "candidate_level",
                                run,
                                aggregation=aggregation,
                                cost_model=cost_model,
                                min_segment_size=minimum,
                                result=result,
                            )
                        )

        # Alternative BIC that also counts the K-1 discrete breakpoint locations.
        try:
            result = identify_working_point(
                candidates,
                penalty_mode="segment_plus_breakpoints",
            )
        except ValueError as exc:
            rows.append(
                sensitivity_row(
                    "alternative_bic_penalty",
                    run,
                    penalty_mode="segment_plus_breakpoints",
                    result=None,
                    status=str(exc),
                )
            )
        else:
            rows.append(
                sensitivity_row(
                    "alternative_bic_penalty",
                    run,
                    penalty_mode="segment_plus_breakpoints",
                    result=result,
                )
            )

        # One representative median CV per non-empty 5 V bin removes local
        # candidate-density weighting introduced by adaptive sequential sampling.
        try:
            binned = identify_binned_working_point(candidates, bin_width_v=5.0)
        except ValueError as exc:
            rows.append(
                sensitivity_row(
                    "five_volt_bins",
                    run,
                    bin_width_v=5,
                    result=None,
                    status=str(exc),
                )
            )
        else:
            row = sensitivity_row("five_volt_bins", run, bin_width_v=5)
            row.update(binned)
            rows.append(row)

    # Diagnostic only: retain the 635 V point in all main analyses, but remove
    # it once to determine whether it alone drives the long-budget-2 Gaussian result.
    if "长预算2" in runs:
        loo_candidates = [candidate for candidate in runs["长预算2"] if candidate.uc != 635.0]
        for cost_model in COST_MODELS:
            try:
                result = identify_working_point(loo_candidates, cost_model=cost_model)
            except ValueError as exc:
                rows.append(
                    sensitivity_row(
                        "leave_one_out_635V",
                        "长预算2",
                        cost_model=cost_model,
                        excluded_uc_v=635,
                        result=None,
                        status=str(exc),
                    )
                )
            else:
                rows.append(
                    sensitivity_row(
                        "leave_one_out_635V",
                        "长预算2",
                        cost_model=cost_model,
                        excluded_uc_v=635,
                        result=result,
                    )
                )
    return rows


def run_prefix_analysis(runs: dict[str, list[Candidate]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run, candidates in runs.items():
        if len(candidates) < 60:
            continue
        for prefix_n in (30, 40, 50, 60):
            result = identify_working_point(candidates, prefix_n=prefix_n)
            rows.append(result_row(result, prefix_n=prefix_n))
    return rows


def segment_rows(results: Sequence[WorkingPointResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for segment_number, segment in enumerate(result.segments, start=1):
            rows.append(
                {
                    "run": result.run,
                    "selected_K": result.segment_count,
                    "segment_number": segment_number,
                    "index_start_1based": segment.start + 1,
                    "index_stop_1based": segment.stop,
                    "candidate_count": segment.count,
                    "uc_min_v": segment.uc_min,
                    "uc_max_v": segment.uc_max,
                    "median_cv_percent": segment.median_cv,
                    "median_log_cv": segment.median_log_cv,
                    "state": segment.state,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify the lowest-score measured Uc in the terminal low-CV branch"
    )
    parser.add_argument("input", type=Path, help="candidate_data.csv")
    parser.add_argument("--output-json", type=Path, default=Path("stable_working_point_results.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("stable_working_point_results.csv"))
    parser.add_argument("--sensitivity-csv", type=Path, default=Path("stable_working_point_sensitivity.csv"))
    parser.add_argument("--prefix-csv", type=Path, default=Path("stable_working_point_prefix.csv"))
    parser.add_argument("--segments-csv", type=Path, default=Path("stable_working_point_segments.csv"))
    parser.add_argument("--aggregation", choices=AGGREGATIONS, default="closest_pair")
    parser.add_argument("--cost-model", choices=COST_MODELS, default="laplace")
    parser.add_argument("--min-segment-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs, excluded = load_candidates(args.input)
    if not runs:
        raise SystemExit("No valid three-shot candidates were found")
    results = [
        identify_working_point(
            candidates,
            aggregation=args.aggregation,
            min_segment_size=args.min_segment_size,
            cost_model=args.cost_model,
        )
        for candidates in runs.values()
    ]
    write_csv(args.output_csv, (result_row(result) for result in results))
    write_csv(args.segments_csv, segment_rows(results))
    sensitivity = run_sensitivity(runs)
    write_csv(args.sensitivity_csv, sensitivity)
    prefixes = run_prefix_analysis(runs)
    if prefixes:
        write_csv(args.prefix_csv, prefixes)
    payload = {
        "method": {
            "aggregation": args.aggregation,
            "cost_model": args.cost_model,
            "min_segment_size": args.min_segment_size,
            "penalty_mode": "segment_parameters",
            "log_cv_epsilon": EPS,
            "cv_input_unit": "percent value",
            "selection": "minimum measured candidate score in terminal low-CV branch",
        },
        "excluded_rows": excluded,
        "results": [asdict(result) for result in results],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(
            f"{result.run}: transition=[{result.transition_lower_uc:g}, "
            f"{result.transition_upper_uc:g}] V, stable start={result.stable_start_uc:g} V, "
            f"Uc*={result.optimum_uc:g} V, score={result.optimum_score:.4f}"
        )
    if excluded:
        print(f"Excluded rows without three valid numeric shots: {len(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
