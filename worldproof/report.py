"""Report / aggregation layer: turn per-rollout metric results into the product.

This is the report-layer half of the two-layer design (SPEC §4, §6.6). It
aggregates a :class:`~worldproof.metrics.RunReport` (per-rollout `MetricResult`s
over a folder of rollouts) into robust cross-rollout statistics and renders them:

- **Robust aggregation, not point estimates** (Agarwal et al. 2021): each
  metric's per-rollout scores are summarized by the interquartile mean (IQM)
  with a bootstrap confidence interval (seeded → reproducible), plus a
  performance profile (score distribution). Never bare mean±std.
- **Latency/cost** columns from `RolloutMetadata.inference_latency_s`.
- **Transparency**: which metrics were skipped and why (the anti-VBench point).
- **Outputs**: a JSON blob for CI and a single self-contained HTML report card
  (inline-SVG horizon curves, worst-N clips, plain-language verdict) — both
  produced on the laptop-only evaluate path, core deps only.

The set-level measures — FVD and the action-recoverability probe — live in
``worldproof.metrics.aggregate`` and are run here: they consume the whole set of
rollouts (not one), so they are a report-layer consumer, skip-and-reported like
the per-rollout metrics.
"""

from __future__ import annotations

import base64
import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO

import numpy as np

from worldproof.core import Rollout
from worldproof.metrics.aggregate import (
    AggregateResult,
    default_aggregates,
    run_aggregates,
)
from worldproof.metrics.base import Metric
from worldproof.metrics.runner import Capabilities, MetricRunner, RunReport

__all__ = [
    "MetricSummary",
    "Report",
    "interquartile_mean",
    "bootstrap_ci",
    "performance_profile",
    "default_metrics",
    "evaluate",
    "build_report",
    "report_json",
    "report_html",
]


# --------------------------------------------------------------------------- #
# Robust aggregation
# --------------------------------------------------------------------------- #


def interquartile_mean(values: Sequence[float]) -> float:
    """Mean of the middle 50% of ``values`` (trims 25% from each tail)."""
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_values.size
    if n == 0:
        return float("nan")
    if n < 4:
        return float(sorted_values.mean())
    trim = n // 4
    return float(sorted_values[trim : n - trim].mean())


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the interquartile mean (seeded → reproducible).

    Plain resampling over rollouts; when task/strata labels exist this becomes a
    stratified bootstrap (resampling within strata) — deferred until tasks are
    labeled.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = interquartile_mean(rng.choice(array, size=array.size, replace=True))
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha)))


def performance_profile(
    values: Sequence[float], thresholds: Sequence[float], *, higher_is_better: bool
) -> list[float]:
    """Fraction of runs that beat each threshold (a score distribution curve)."""
    array = np.asarray(values, dtype=np.float64)
    if higher_is_better:
        return [float((array >= t).mean()) for t in thresholds]
    return [float((array <= t).mean()) for t in thresholds]


# --------------------------------------------------------------------------- #
# Report data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricSummary:
    """Cross-rollout aggregate for one metric."""

    name: str
    higher_is_better: bool
    iqm: float
    ci_low: float
    ci_high: float
    mean: float
    median: float
    n_rollouts: int
    mean_curve: tuple[float, ...]  # horizon curve averaged over rollouts
    profile_thresholds: tuple[float, ...]
    profile: tuple[float, ...]


@dataclass(frozen=True)
class Report:
    """The evaluate-path report: robust aggregates + provenance + verdict."""

    metrics: tuple[MetricSummary, ...]
    n_rollouts: int
    device: str
    latency: Mapping[str, float] | None
    skips: Mapping[str, Mapping[str, object]]
    verdict: str
    metric_versions: Mapping[str, str]
    aggregates: tuple[AggregateResult, ...] = ()


# --------------------------------------------------------------------------- #
# Building the report
# --------------------------------------------------------------------------- #


def default_metrics() -> list[Metric]:
    """All registered per-rollout metrics, instantiated with defaults."""
    from worldproof.metrics.registry import REGISTRY

    return [REGISTRY.create(name) for name in REGISTRY.names()]


def evaluate(
    rollouts: Sequence[Rollout],
    *,
    metrics: Sequence[Metric] | None = None,
    capabilities: Capabilities | None = None,
    seed: int = 0,
) -> tuple[Report, RunReport]:
    """Run metrics over ``rollouts`` and build the aggregated report.

    Every environment capability travels through ``capabilities``: the tracker
    for the invariants, and the FVD video feature extractor
    (``Capabilities.detect(fvd_extractor=...)``, e.g.
    ``default_video_extractor()``); with no extractor FVD skips-and-reports.
    """
    runner = MetricRunner(
        list(metrics) if metrics is not None else default_metrics(),
        capabilities=capabilities,
    )
    run_report = runner.run_many(list(rollouts))
    report = build_report(
        rollouts, run_report, seed=seed, capabilities=runner.capabilities
    )
    return report, run_report


def build_report(
    rollouts: Sequence[Rollout],
    run_report: RunReport,
    *,
    seed: int = 0,
    n_boot: int = 2000,
    capabilities: Capabilities | None = None,
) -> Report:
    rollouts = list(rollouts)
    by_metric: dict[str, list] = {}
    for run in run_report.runs:
        for result in run.results:
            by_metric.setdefault(result.name, []).append(result)

    summaries: list[MetricSummary] = []
    for name in sorted(by_metric):
        results = by_metric[name]
        scores = np.array([r.summary for r in results], dtype=np.float64)
        higher_is_better = results[0].higher_is_better
        min_horizon = min(r.horizon for r in results)
        mean_curve = np.stack([r.mean_curve[:min_horizon] for r in results]).mean(
            axis=0
        )
        thresholds = np.linspace(float(scores.min()), float(scores.max()), 11)
        ci_low, ci_high = bootstrap_ci(scores, seed=seed, n_boot=n_boot)
        summaries.append(
            MetricSummary(
                name=name,
                higher_is_better=higher_is_better,
                iqm=interquartile_mean(scores),
                ci_low=ci_low,
                ci_high=ci_high,
                mean=float(scores.mean()),
                median=float(np.median(scores)),
                n_rollouts=len(results),
                mean_curve=tuple(float(v) for v in mean_curve),
                profile_thresholds=tuple(float(t) for t in thresholds),
                profile=tuple(
                    performance_profile(
                        scores, thresholds, higher_is_better=higher_is_better
                    )
                ),
            )
        )

    measures = default_aggregates(
        fvd_extractor=capabilities.fvd_extractor if capabilities else None
    )
    aggregate_results, aggregate_skips = run_aggregates(rollouts, measures, seed=seed)

    # One merge rule for both layers: count every skip occurrence, keep the
    # first-seen reason. No entry is ever silently dropped.
    skips: dict[str, dict[str, object]] = {}
    for skip in [*run_report.all_skips, *aggregate_skips]:
        entry = skips.setdefault(skip.metric, {"count": 0, "reason": skip.reason})
        entry["count"] = int(entry["count"]) + 1  # type: ignore[arg-type]

    # One provenance rule for both layers: record the version of every
    # *configured* measure, whether or not it produced a result.
    aggregate_versions = {m.name: m.version for m in measures}

    latencies = [
        r.metadata.inference_latency_s
        for r in rollouts
        if r.metadata.inference_latency_s is not None
    ]
    latency = None
    if latencies:
        arr = np.asarray(latencies, dtype=np.float64)
        latency = {
            "median_s": float(np.median(arr)),
            "p90_s": float(np.quantile(arr, 0.9)),
            "max_s": float(arr.max()),
            "n": float(len(latencies)),
        }

    report = Report(
        metrics=tuple(summaries),
        n_rollouts=len(rollouts),
        device=run_report.device,
        latency=latency,
        skips=skips,
        verdict="",
        metric_versions={**dict(run_report.metric_versions), **aggregate_versions},
        aggregates=tuple(aggregate_results),
    )
    return replace(report, verdict=_verdict(report))


def _verdict(report: Report) -> str:
    n_rollouts = report.n_rollouts
    device = report.device
    summaries = report.metrics
    skips = report.skips
    latency = report.latency
    aggregates = report.aggregates
    lines = [
        f"Evaluated {n_rollouts} rollout(s) on {len(summaries)} metric(s) "
        f"(device: {device})."
    ]
    for summary in summaries:
        direction = "higher=better" if summary.higher_is_better else "lower=better"
        lines.append(
            f"  {summary.name}: IQM {summary.iqm:.4g} "
            f"[{summary.ci_low:.4g}, {summary.ci_high:.4g}] ({direction})"
        )
    for aggregate in aggregates:
        direction = "higher=better" if aggregate.higher_is_better else "lower=better"
        lines.append(
            f"  {aggregate.name}: {aggregate.summary:.4g} (set-level, {direction})"
        )
    if skips:
        skipped = ", ".join(
            f"{name} (x{entry['count']})" for name, entry in skips.items()
        )
        lines.append(f"Skipped (unsupported here): {skipped}.")
    if latency is not None:
        lines.append(
            f"Inference latency: median {latency['median_s']:.3g}s, "
            f"p90 {latency['p90_s']:.3g}s."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def report_json(report: Report) -> dict:
    """A plain, CI-friendly dict of the report (JSON-serializable)."""
    return {
        "n_rollouts": report.n_rollouts,
        "device": report.device,
        "metrics": [
            {
                "name": m.name,
                "higher_is_better": m.higher_is_better,
                "iqm": m.iqm,
                "ci": [m.ci_low, m.ci_high],
                "mean": m.mean,
                "median": m.median,
                "n_rollouts": m.n_rollouts,
                "mean_curve": list(m.mean_curve),
                "performance_profile": {
                    "thresholds": list(m.profile_thresholds),
                    "fraction": list(m.profile),
                },
            }
            for m in report.metrics
        ],
        "aggregates": [
            {
                "name": a.name,
                "higher_is_better": a.higher_is_better,
                "summary": _finite_or_none(a.summary),
                "curve": (
                    [_finite_or_none(v) for v in a.curve]
                    if a.curve is not None
                    else None
                ),
                "n_items": a.n_items,
                "extra": dict(a.extra),
            }
            for a in report.aggregates
        ],
        "latency": dict(report.latency) if report.latency else None,
        "skips": {k: dict(v) for k, v in report.skips.items()},
        "metric_versions": dict(report.metric_versions),
        "verdict": report.verdict,
    }


def _finite_or_none(value: float) -> float | None:
    """Map non-finite floats (nan / inf) to ``None`` so JSON stays strict."""
    return float(value) if math.isfinite(value) else None


# --------------------------------------------------------------------------- #
# Self-contained HTML report card
# --------------------------------------------------------------------------- #


def _svg_curve(values: Sequence[float], *, width: int = 340, height: int = 104) -> str:
    """A small line chart of a per-step curve, with value and step axes.

    The y axis is labeled with the value range (max at top, min at bottom) so the
    auto-scaled shape is readable in absolute terms; the x axis marks the horizon
    steps; and each step is a dot whose hover title gives the exact value.
    """
    vals = list(values)
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    pad_l, pad_r, pad_t, pad_b = 52, 8, 10, 16
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    points, dots = [], []
    for i, v in enumerate(vals):
        x = pad_l + (i / max(n - 1, 1)) * plot_w
        y = pad_t + (1 - (v - lo) / span) * plot_h
        points.append(f"{x:.1f},{y:.1f}")
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="#2a7d5f">'
            f"<title>step {i + 1}: {v:.4g}</title></circle>"
        )

    axis = (
        f'<text x="{pad_l - 4}" y="{pad_t + 4}" font-size="9" fill="#888" '
        f'text-anchor="end">{hi:.3g}</text>'
        f'<text x="{pad_l - 4}" y="{pad_t + plot_h}" font-size="9" fill="#888" '
        f'text-anchor="end">{lo:.3g}</text>'
        f'<text x="{pad_l}" y="{height - 4}" font-size="9" fill="#888" '
        f'text-anchor="start">1</text>'
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 4}" font-size="9" '
        f'fill="#aaa" text-anchor="middle">step</text>'
        f'<text x="{pad_l + plot_w:.1f}" y="{height - 4}" font-size="9" fill="#888" '
        f'text-anchor="end">{n}</text>'
    )
    return (
        f'<svg width="{width}" height="{height}" '
        f'style="background:#fafafa;border:1px solid #ddd">'
        f'<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="#2a7d5f" stroke-width="2"/>{"".join(dots)}{axis}</svg>'
    )


def _clip_data_uri(frames: np.ndarray) -> str:
    from PIL import Image

    images = []
    for frame in frames:
        if frame.shape[-1] == 1:
            images.append(Image.fromarray(frame[..., 0], mode="L"))
        else:
            images.append(Image.fromarray(frame, mode="RGB"))
    buffer = BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{encoded}"


_CLIP_RANKING_PREFERENCE = ("psnr", "ssim", "lpips")


def _worst_rollout_indices(
    report: Report, run_report: RunReport, rollouts: Sequence[Rollout], n: int
) -> list[int]:
    """Indices of the worst pixel rollouts, ranked by a metric they actually have.

    Only metrics that produced results *on the pixel rollouts* are candidates
    (a latent metric that happens to sort first must never be picked, which
    would silently select no clips). Preference order: psnr, ssim, lpips, then
    any pixel metric, so the ranking is stable across configurations.
    """
    by_metric: dict[str, list[tuple[float, int]]] = {}
    direction: dict[str, bool] = {}
    for run in run_report.runs:
        if rollouts[run.index].modality != "pixels":
            continue
        for result in run.results:
            by_metric.setdefault(result.name, []).append((result.summary, run.index))
            direction[result.name] = result.higher_is_better
    if not by_metric:
        return []
    ranking_name = next(
        (name for name in _CLIP_RANKING_PREFERENCE if name in by_metric),
        sorted(by_metric)[0],
    )
    scored = by_metric[ranking_name]
    # worst = lowest score when higher-is-better, else highest
    scored.sort(reverse=not direction[ranking_name])
    return [idx for _, idx in scored[:n]]


def report_html(
    report: Report,
    rollouts: Sequence[Rollout] | None = None,
    run_report: RunReport | None = None,
    *,
    worst_n: int = 3,
) -> str:
    e = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>worldproof report</title><style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#222;max-width:900px}",
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ddd;"
        "padding:.4rem .8rem;text-align:left}th{background:#f4f4f4}",
        ".verdict{white-space:pre-wrap;background:#f7f9f8;"
        "border-left:3px solid #2a7d5f;padding:1rem;border-radius:4px}"
        ".clip{margin:.5rem;display:inline-block}",
        "</style></head><body>",
        "<h1>worldproof report</h1>",
        f"<div class='verdict'>{e(report.verdict)}</div>",
        "<h2>Metrics</h2><table><tr><th>metric</th><th>IQM</th><th>95% CI</th>"
        "<th>dir</th><th>n</th><th>horizon curve</th></tr>",
    ]
    for m in report.metrics:
        direction = "↑" if m.higher_is_better else "↓"
        parts.append(
            f"<tr><td>{e(m.name)}</td><td>{m.iqm:.4g}</td>"
            f"<td>[{m.ci_low:.4g}, {m.ci_high:.4g}]</td><td>{direction}</td>"
            f"<td>{m.n_rollouts}</td><td>{_svg_curve(m.mean_curve)}</td></tr>"
        )
    parts.append("</table>")

    if report.aggregates:
        parts.append(
            "<h2>Set-level metrics</h2>"
            "<p style='color:#666;font-size:.9em'>Computed across the whole set "
            "(not per-rollout). FVD is a labeled weak reference (quality, not "
            "dynamics); action-recoverability is the primary latent diagnostic "
            "(R&#178; per step).</p>"
            "<table><tr><th>metric</th><th>value</th><th>dir</th><th>n</th>"
            "<th>horizon curve</th></tr>"
        )
        for a in report.aggregates:
            direction = "&#8593;" if a.higher_is_better else "&#8595;"
            value = f"{a.summary:.4g}" if math.isfinite(a.summary) else "n/a"
            finite_curve = [v for v in a.curve if math.isfinite(v)] if a.curve else []
            curve_svg = _svg_curve(finite_curve) if len(finite_curve) > 1 else ""
            parts.append(
                f"<tr><td>{e(a.name)}</td><td>{value}</td><td>{direction}</td>"
                f"<td>{a.n_items}</td><td>{curve_svg}</td></tr>"
            )
        parts.append("</table>")

    if report.latency is not None:
        parts.append(
            "<h2>Latency</h2><table><tr><th>median</th><th>p90</th><th>max</th></tr>"
            f"<tr><td>{report.latency['median_s']:.3g}s</td>"
            f"<td>{report.latency['p90_s']:.3g}s</td>"
            f"<td>{report.latency['max_s']:.3g}s</td></tr></table>"
        )

    if report.skips:
        parts.append(
            "<h2>Skipped metrics</h2><table><tr><th>metric</th><th>why</th></tr>"
        )
        for name, entry in report.skips.items():
            parts.append(
                f"<tr><td>{e(name)}</td><td>{e(str(entry['reason']))}</td></tr>"
            )
        parts.append("</table>")

    if rollouts is not None and run_report is not None:
        indices = _worst_rollout_indices(report, run_report, rollouts, worst_n)
        if indices:
            parts.append("<h2>Worst-N clips (predicted rollouts)</h2>")
            for idx in indices:
                uri = _clip_data_uri(rollouts[idx].predictions[0])
                parts.append(
                    f"<span class='clip'><img src='{uri}' width='128'><br>"
                    f"rollout {idx}</span>"
                )

    parts.append("</body></html>")
    return "".join(parts)
