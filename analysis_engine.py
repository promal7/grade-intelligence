"""
analysis_engine.py
-------------------
Three jobs, matching the three graded deliverables directly:

1. find_correlations()   -> "Find new correlations not defined in the system"
2. assess_trajectory_risk() -> "Predict when the spec is at risk of going off-spec
                                 during transition ... before quality limits exceeded"
3. generate_recommendations() -> "Recommendation of setpoints to keep system in
                                   safe operating limits" + "Tag every suggestion
                                   with possible source of inference"

Deliberately NOT a black-box ML model. Everything here is an interpretable
statistic (Pearson/Spearman correlation, percentile envelopes, linear
trajectory extrapolation) so every recommendation can point at a number and
say exactly where it came from -- which is what the "rationale" and
"source of inference" requirements are actually asking for.
"""

import numpy as np
import pandas as pd
from scipy import stats

DRIVER_COLS = ["steam_ramp_rate", "filler_step_size", "speed_delta", "moisture_lag_steps",
               "stock_flow_volatility", "ash_volatility", "caliper_deviation"]
TARGET_COLS = ["max_deviation_pct", "stabilization_time"]
SPEC_LIMIT_PCT = 2.5

# Fixed process/recipe limits -- distinct from safe_envelope() below, which is
# learned from historical data. These represent hard engineering ceilings a
# recipe management system would enforce regardless of what history shows.
# ILLUSTRATIVE VALUES: we don't have access to Honeywell's real recipe
# management system, so these are reasonable placeholders in the same units
# as the simulated data, not real mill specifications. A production version
# would read these from the actual recipe database instead of this constant.
RECIPE_LIMITS = {
    "steam_ramp_rate": 0.85,
    "filler_step_size": 8.5,
    "speed_delta": 6.0,
    "moisture_lag_steps": 10,
}


def load_data(meta_path="events_meta.csv", ts_path="events_timeseries.csv"):
    meta = pd.read_csv(meta_path)
    ts = pd.read_csv(ts_path)
    return meta, ts


# ---------------------------------------------------------------------------
# 1. Correlation discovery
# ---------------------------------------------------------------------------
def find_correlations(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Correlates each candidate driver variable against each outcome metric.
    This is the 'new correlations found by the solution' the dashboard must show.
    """
    rows = []
    for driver in DRIVER_COLS:
        for target in TARGET_COLS:
            x, y = meta[driver].values, meta[target].values
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            rows.append({
                "driver": driver, "target": target,
                "pearson_r": round(r_p, 3), "pearson_p": round(p_p, 4),
                "spearman_r": round(r_s, 3), "spearman_p": round(p_s, 4),
                "n_events": len(meta),
                "strength": "strong" if abs(r_p) >= 0.5 else ("moderate" if abs(r_p) >= 0.3 else "weak"),
            })
    out = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2. Safe operating envelope (learned from historically successful events)
# ---------------------------------------------------------------------------
def safe_envelope(meta: pd.DataFrame, percentile=75) -> pd.DataFrame:
    """
    For each driver, what value did GOOD (went_offspec == False) transitions
    stay under? That's the empirical 'safe zone' used to generate recommendations.
    """
    good = meta[meta["went_offspec"] == False]
    bad = meta[meta["went_offspec"] == True]
    rows = []
    for driver in DRIVER_COLS:
        safe_bound = float(np.percentile(good[driver], percentile)) if len(good) else float(meta[driver].median())
        rows.append({
            "driver": driver,
            "safe_upper_bound": round(safe_bound, 3),
            "good_median": round(float(good[driver].median()), 3) if len(good) else None,
            "bad_median": round(float(bad[driver].median()), 3) if len(bad) else None,
            "n_good": len(good), "n_bad": len(bad),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Trajectory extrapolation / risk assessment (for an in-progress transition)
# ---------------------------------------------------------------------------
def assess_trajectory_risk(partial_ts: pd.DataFrame, bw_target: float,
                            current_t: int, lookback=8, project_ahead=15):
    """
    Given a partial time series up to `current_t`, fit a simple linear trend
    to the last `lookback` basis_weight points and project forward. Flags
    risk if the projected trajectory implies the process won't be within
    spec by the time it should have stabilized.
    """
    window = partial_ts[partial_ts["t"] <= current_t].tail(lookback)
    if len(window) < 3:
        return {"risk": "insufficient_data", "projected_max_dev_pct": None,
                "projected_path": None, "slope": None}

    x = window["t"].values
    y = window["basis_weight"].values
    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    future_t = np.arange(current_t, current_t + project_ahead)
    projected = intercept + slope * future_t
    projected_dev_pct = 100 * np.abs(projected - bw_target) / bw_target
    max_proj_dev = float(np.max(projected_dev_pct))

    current_dev_pct = 100 * abs(y[-1] - bw_target) / bw_target
    trend_is_worsening = abs(slope) > 0.02 and np.sign(y[-1] - bw_target) == np.sign(slope) if slope != 0 else False

    if current_dev_pct > SPEC_LIMIT_PCT and trend_is_worsening:
        risk = "high"
    elif max_proj_dev > SPEC_LIMIT_PCT:
        risk = "medium"
    else:
        risk = "low"

    return {
        "risk": risk,
        "current_dev_pct": round(current_dev_pct, 3),
        "projected_max_dev_pct": round(max_proj_dev, 3),
        "projected_path": list(zip(future_t.tolist(), projected.tolist())),
        "slope": round(float(slope), 4),
        "trend_r2": round(float(r_value ** 2), 3),
    }


# ---------------------------------------------------------------------------
# Stabilization-time impact estimate (closes the "reduce stabilization time"
# requirement -- turns a correlation into an actual predicted number)
# ---------------------------------------------------------------------------
def estimate_stabilization_impact(meta: pd.DataFrame, driver: str, current_val: float, target_val: float):
    """
    Fits stabilization_time ~ driver across historical events, then reads off
    the predicted change in stabilization time if this driver moved from
    current_val to target_val. This is what lets a recommendation say
    "~N fewer steps to stabilize" instead of just "this driver correlates."
    """
    x = meta[driver].values
    y = meta["stabilization_time"].values
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    predicted_current = intercept + slope * current_val
    predicted_target = intercept + slope * target_val
    delta = predicted_current - predicted_target  # positive = improvement (fewer steps)
    return {
        "delta_steps": round(float(delta), 1),
        "r2": round(float(r_value ** 2), 3),
        "p_value": round(float(p_value), 4),
        "slope": round(float(slope), 4),
    }


# ---------------------------------------------------------------------------
# Recommendation generation (rationale-tagged, per deliverable #5)
# ---------------------------------------------------------------------------
def generate_recommendations(current_drivers: dict, envelope: pd.DataFrame,
                              correlations: pd.DataFrame, risk: dict, meta: pd.DataFrame = None):
    """
    current_drivers: dict like {"steam_ramp_rate": 0.7, "filler_step_size": 6.2, ...}
    Returns a list of recommendation dicts, each tagged with WHY it's being made.
    meta: full historical dataframe, needed for the stabilization-time delta estimate.
    """
    recs = []
    env_lookup = envelope.set_index("driver").to_dict("index")

    for driver, current_val in current_drivers.items():
        if driver not in env_lookup:
            continue

        recipe_limit = RECIPE_LIMITS.get(driver)
        safe_bound = env_lookup[driver]["safe_upper_bound"]

        # Recipe limit takes priority: a hard-engineering-ceiling breach is a
        # different (and more certain) kind of finding than a statistical one.
        if recipe_limit is not None and current_val > recipe_limit:
            target_val = recipe_limit
            source = "recipe"
            confidence = "high"
            rationale = (f"{driver.replace('_', ' ').title()} of {current_val:.2f} exceeds the fixed "
                         f"recipe limit of {recipe_limit:.2f} for this loop, independent of what "
                         f"historical data shows.")
        elif current_val > safe_bound:
            target_val = safe_bound
            source = "historical_correlation"
            rel = correlations[correlations["driver"] == driver].iloc[0]
            confidence = "high" if abs(rel["pearson_r"]) >= 0.5 and rel["pearson_p"] < 0.01 else \
                         ("medium" if abs(rel["pearson_r"]) >= 0.3 else "low")
            rationale = (f"{driver.replace('_', ' ').title()} correlates with "
                        f"{rel['target'].replace('_', ' ')} at r={rel['pearson_r']} "
                        f"(p={rel['pearson_p']}, n={rel['n_events']} historical events).")
        else:
            continue  # within both recipe limit and historical safe envelope

        text = (f"Reduce {driver.replace('_', ' ')} from {current_val:.2f} toward {target_val:.2f}")
        if source == "historical_correlation":
            text += " (the 75th-percentile value seen in historically successful transitions)."
        else:
            text += " (the recipe-defined ceiling for this loop)."

        # Attach the stabilization-time impact estimate whenever we have the
        # historical data to fit it -- this is what makes "reduce stabilization
        # time" an actual output instead of an implied side effect.
        if meta is not None:
            impact = estimate_stabilization_impact(meta, driver, current_val, target_val)
            if impact["delta_steps"] > 0.5 and impact["p_value"] < 0.05:
                text += (f" Estimated to cut stabilization time by ~{impact['delta_steps']} steps "
                         f"(linear fit, R\u00b2={impact['r2']}, p={impact['p_value']}).")

        recs.append({
            "id": f"rec_{driver}",
            "text": text,
            "rationale": rationale,
            "source_of_inference": source,
            "confidence": confidence,
        })

    if risk.get("risk") in ("high", "medium"):
        recs.append({
            "id": "rec_trajectory_alert",
            "text": (f"Current trajectory projects {risk['projected_max_dev_pct']}% max "
                     f"deviation, exceeding the {SPEC_LIMIT_PCT}% spec limit. "
                     f"Consider holding current ramp rates rather than accelerating further."),
            "rationale": (f"Linear extrapolation of the last data points "
                          f"(trend R\u00b2={risk.get('trend_r2')}, slope={risk.get('slope')}) "
                          f"projects a spec breach before nominal stabilization."),
            "source_of_inference": "trajectory_extrapolation",
            "confidence": "high" if risk.get("trend_r2", 0) > 0.6 else "medium",
        })

    return recs


if __name__ == "__main__":
    meta, ts = load_data()
    corr = find_correlations(meta)
    print("=== Correlations ===")
    print(corr.to_string(index=False))

    env = safe_envelope(meta)
    print("\n=== Safe envelope ===")
    print(env.to_string(index=False))

    sample_event = ts[ts.event_id == 3]
    sample_meta = meta[meta.event_id == 3].iloc[0]
    risk = assess_trajectory_risk(sample_event, sample_meta["bw_target"], current_t=20)
    print("\n=== Sample trajectory risk (event 3, t=20) ===")
    print(risk)

    current_drivers = {c: sample_meta[c] for c in DRIVER_COLS}
    recs = generate_recommendations(current_drivers, env, corr, risk, meta=meta)
    print("\n=== Sample recommendations ===")
    for r in recs:
        print(r)
