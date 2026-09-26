"""Paired quality evidence with explicit, separately gated scientific claims.

Positive advantage always favors OpenNN. Matching seed labels define blocks;
they do not imply that the engines used identical initial parameter tensors.
Intervals describe variation across seeds on one fixed held-out dataset, not
uncertainty over new datasets. Historical inputs remain descriptive by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = {
    "dense": ("accuracy_percent", "higher"),
    "lstm": ("rmse", "lower"),
    "cnn": ("accuracy_percent", "higher"),
    "transformer": ("bleu", "higher"),
}
ENGINES = ("opennn", "pytorch")


def _beta_fraction(a, b, x):
    """Continued fraction for the regularized incomplete beta function."""
    tiny = 1e-300
    c = 1.0
    d = 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    value = d
    for index in range(1, 1001):
        for coefficient in (
            index * (b - index) * x / ((a + 2 * index - 1) * (a + 2 * index)),
            -(a + index) * (a + b + index) * x
            / ((a + 2 * index) * (a + 2 * index + 1)),
        ):
            d = 1 + coefficient * d
            c = 1 + coefficient / c
            d = 1 / (d if abs(d) > tiny else tiny)
            c = c if abs(c) > tiny else tiny
            delta = d * c
            value *= delta
        if abs(delta - 1) < 3e-14:
            return value
    raise ArithmeticError("Incomplete beta continued fraction did not converge")


def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_fraction(a, b, x) / a
    return 1 - factor * _beta_fraction(b, a, 1 - x) / b


def student_t_critical(probability, degrees_of_freedom):
    """Student-t quantile, using only the Python standard library."""
    if not 0.5 < probability < 1 or degrees_of_freedom < 1:
        raise ValueError("Require 0.5 < probability < 1 and positive degrees of freedom")
    target = 1 - probability

    def survival(value):
        x = degrees_of_freedom / (degrees_of_freedom + value * value)
        return 0.5 * _regularized_beta(x, degrees_of_freedom / 2, 0.5)

    lower, upper = 0.0, 1.0
    while survival(upper) > target:
        upper *= 2
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if survival(midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def _interval(mean, standard_error, count, confidence):
    radius = student_t_critical((1 + confidence) / 2, count - 1) * standard_error
    return [mean - radius, mean + radius]


def _seeds(values):
    values = list(values)
    if any(type(seed) is not int or seed < 0 for seed in values):
        raise ValueError("Seeds must be nonnegative integers")
    if len(set(values)) != len(values) or not values:
        raise ValueError("Seeds must be nonempty and distinct")
    return values


def _pair_records(records, model, metric, planned_seeds):
    grouped = {}
    for record in records:
        if record["model"] == model:
            grouped.setdefault(record["seed"], {}).setdefault(
                record["engine"], []
            ).append(record)
    expected = set(planned_seeds or grouped)
    pairs, issues = [], []
    for seed in sorted(set(grouped) | expected):
        engines = grouped.get(seed, {})
        seed_issues = []
        scores = {}
        for engine in ENGINES:
            runs = engines.get(engine, [])
            if not runs:
                seed_issues.append({"engine": engine, "reason": "missing"})
            elif len(runs) != 1:
                seed_issues.append({"engine": engine, "reason": "duplicate"})
            elif runs[0].get("status") != "ok":
                seed_issues.append({"engine": engine, "reason": "failed"})
            else:
                value = runs[0].get("metrics", {}).get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(value):
                    seed_issues.append({"engine": engine, "reason": "invalid_metric"})
                else:
                    scores[engine] = float(value)
        if not seed_issues:
            native = engines["opennn"][0].get("driver", {})
            torch = engines["pytorch"][0].get("driver", {})
            for field in ("device", "precision", "epochs", "parameters", "train_samples", "test_samples"):
                if native.get(field) != torch.get(field):
                    seed_issues.append({"reason": "driver_mismatch", "field": field})
        if seed_issues:
            issues.append({"seed": seed, "issues": seed_issues})
        else:
            pairs.append({"seed": seed, **scores})
    return pairs, issues, sorted(set(grouped) - expected)


def _claim(reasons, supported):
    return {
        "status": "unassessed" if reasons else (
            "supported" if supported else "not_established"
        ),
        "established": not reasons and bool(supported),
        "reasons": reasons,
    }


def analyze_pairs(records, *, planned_seeds=None, models=None, margins=None,
                  confirmatory=False, protocol_predeclared=False,
                  minimum_quality_reviewed=False, metric_specs=None):
    """Analyze all valid observed pairs without dropping unfavorable outcomes.

    Claim eligibility must be attested by a caller enforcing a frozen protocol;
    these booleans cannot themselves verify preregistration. A symmetric margin
    uses original metric units and must be positive. No margin is inferred.
    Zero sample variance makes t inference unavailable, not automatic parity.
    """
    records = list(records)
    specs = dict(METRICS)
    specs.update(metric_specs or {})
    if planned_seeds is not None:
        planned_seeds = _seeds(planned_seeds)
    observed_models = list(dict.fromkeys(record["model"] for record in records))
    models = list(observed_models if models is None else models)
    if len(set(models)) != len(models) or set(observed_models) - set(models):
        raise ValueError("Model plan must be distinct and include every observed model")
    for model in models:
        if model not in specs or specs[model][1] not in ("higher", "lower"):
            raise ValueError(f"Missing or invalid metric definition for {model}")
    for record in records:
        _seeds([record["seed"]])
        if record["engine"] not in ENGINES:
            raise ValueError(f"Unknown engine: {record['engine']}")
    margins = dict(margins or {})
    if set(margins) - set(models):
        raise ValueError("Margins contain models outside the plan")
    for margin in margins.values():
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) \
                or not math.isfinite(margin) or margin <= 0:
            raise ValueError("Margins must be finite positive values in metric units")

    rows = []
    for model in models:
        metric, direction = specs[model]
        sign = 1 if direction == "higher" else -1
        pairs, issues, unplanned = _pair_records(records, model, metric, planned_seeds)
        for pair in pairs:
            pair["difference"] = pair["opennn"] - pair["pytorch"]
            pair["advantage"] = sign * pair["difference"]
        values = [pair["advantage"] for pair in pairs]
        count = len(values)
        mean = statistics.mean(values) if count else None
        spread = statistics.stdev(values) if count > 1 else None
        inferential = spread is not None and spread > 0
        row = {
            "model": model, "metric": metric, "direction": direction,
            "complete_pairs": count, "pairs": pairs, "pair_issues": issues,
            "unplanned_seeds": unplanned,
            "opennn_mean": statistics.mean(p["opennn"] for p in pairs) if pairs else None,
            "pytorch_mean": statistics.mean(p["pytorch"] for p in pairs) if pairs else None,
            "difference_mean": sign * mean if count else None,
            "advantage_mean": mean, "advantage_stddev": spread,
            "advantage_ci90": None, "advantage_ci95": None,
            "advantage_familywise_ci95": None,
            "margin": margins.get(model),
            "observed_deficit": mean < 0 if count else None,
        }
        if inferential:
            error = spread / math.sqrt(count)
            row["advantage_ci90"] = _interval(mean, error, count, 0.90)
            row["advantage_ci95"] = _interval(mean, error, count, 0.95)
            row["advantage_familywise_ci95"] = _interval(
                mean, error, count, 1 - 0.05 / len(models)
            )
        reasons = []
        for condition, reason in (
            (not confirmatory, "descriptive_analysis"),
            (not protocol_predeclared, "protocol_not_predeclared"),
            (not minimum_quality_reviewed, "achieved_quality_not_reviewed"),
            (planned_seeds is None, "seed_plan_missing"),
            (bool(issues), "incomplete_or_invalid_pairs"),
            (bool(unplanned), "unplanned_seeds"),
            (count < 2, "insufficient_pairs"),
            (spread == 0, "zero_variance_t_inference_unavailable"),
            (any(r.get("smoke", False) for r in records if r["model"] == model),
             "synthetic_smoke_data"),
        ):
            if condition:
                reasons.append(reason)
        ci90 = row["advantage_ci90"]
        adjusted = row["advantage_familywise_ci95"]
        margin = row["margin"]
        margin_reasons = reasons + (["margin_missing"] if margin is None else [])
        row["claims"] = {
            "equivalence": _claim(margin_reasons, ci90 is not None and margin is not None
                                  and ci90[0] > -margin and ci90[1] < margin),
            "noninferiority": _claim(margin_reasons, ci90 is not None and margin is not None
                                     and ci90[0] > -margin),
            "superiority": _claim(reasons, adjusted is not None and adjusted[0] > 0),
        }
        ci95 = row["advantage_ci95"]
        row["descriptive_evidence"] = (
            "interval_unavailable" if ci95 is None else
            "interval_below_zero" if ci95[1] < 0 else
            "interval_above_zero" if ci95[0] > 0 else "interval_includes_zero"
        )
        rows.append(row)
    return {
        "schema": "quality-paired-statistics-v1",
        "analysis": "confirmatory_requested" if confirmatory else "descriptive",
        "method": "Paired Student-t intervals across seed blocks; positive advantage favors OpenNN",
        "limitations": [
            "Pairing seed labels does not establish shared initial tensors",
            "Intervals assume independent approximately normal seed differences",
            "Fixed held-out data; intervals do not cover generalization to new datasets",
            "CI90 gives alpha=.05 TOST equivalence and one-sided noninferiority bounds",
            "Superiority uses two-sided CI95 with Bonferroni correction across all planned models",
            "All-model equivalence/noninferiority are conjunctions, not claims on unmeasured tasks",
        ],
        "planned_seeds": planned_seeds,
        "models": models,
        "rows": rows,
        "all_models_equivalent": bool(rows) and all(
            row["claims"]["equivalence"]["established"] for row in rows
        ),
        "all_models_noninferior": bool(rows) and all(
            row["claims"]["noninferiority"]["established"] for row in rows
        ),
    }


def analyze_bundle(bundle, *, margins=None):
    """Analyze quality-v1 or the historical scores_by_seed bundle descriptively."""
    if "runs" in bundle:
        arguments = bundle.get("provenance", {}).get("arguments", {})
        seeds = arguments.get("seeds")
        if isinstance(seeds, str):
            seeds = [int(value) for value in seeds.split(",")]
        models = bundle.get("provenance", {}).get("manifests")
        return analyze_pairs(bundle["runs"], planned_seeds=seeds,
                             models=list(models) if models else None, margins=margins)
    records, specs = [], {}
    for model, evidence in bundle["models"].items():
        specs[model] = (evidence["metric"], evidence["direction"])
        for seed, scores in evidence["scores_by_seed"].items():
            for engine, score in scores.items():
                records.append({"model": model, "engine": engine, "seed": int(seed),
                                "status": "ok", "metrics": {evidence["metric"]: score}})
    return analyze_pairs(records, models=list(specs), margins=margins, metric_specs=specs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margins", type=Path,
                        help="Optional JSON model-to-margin map; remains descriptive")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must not overwrite source evidence")
    source = args.input.read_bytes()
    margins = json.loads(args.margins.read_text()) if args.margins else None
    result = analyze_bundle(json.loads(source), margins=margins)
    result["source"] = {"path": str(args.input.resolve()),
                        "sha256": hashlib.sha256(source).hexdigest()}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
