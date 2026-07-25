"""
data_generator.py
------------------
Generates a SYNTHETIC historical dataset of paper-machine grade-change
transitions. This stands in for real Honeywell QCS/DCS historian data,
which we don't have access to.

Design intent (so this is defensible in the presentation):
- Each "event" is one grade-change transition (old grade -> new grade).
- During a transition, basis weight (the controlled quality variable) moves
  from an old setpoint to a new setpoint. In the real world it overshoots,
  oscillates, and settles depending on how aggressively supporting loops
  (stock flow, filler flow, steam pressure, machine speed) are ramped.
- We inject REALISTIC causal relationships on purpose, so the correlation
  engine in analysis_engine.py has real signal to find (not just noise):
    1. Faster steam pressure ramp rate  -> more overshoot & longer settle time
    2. Larger filler flow step size      -> more ash/caliper disturbance
    3. Higher machine speed delta        -> more basis weight variance
    4. Moisture loop lag                 -> slower stabilization
- Some events are "good" transitions (experienced operator tuning) and some
  are "bad" transitions (aggressive/rushed tuning) -- this gives the model
  both success and failure cases to learn from, per the challenge spec.

Output:
- events_meta.csv      : one row per transition event (summary stats, outcome)
- events_timeseries.csv: one row per (event, timestep) with all process vars
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_EVENTS = 140
N_STEPS = 90  # timesteps per transition (e.g. 1-minute resolution, 90 min window)
SPEC_LIMIT_PCT = 2.5  # off-spec threshold: % deviation from target basis weight

GRADES = ["Grade-A (Light)", "Grade-B (Standard)", "Grade-C (Heavy)", "Grade-D (Specialty)"]
GRADE_TARGET_BW = {"Grade-A (Light)": 45.0, "Grade-B (Standard)": 60.0,
                    "Grade-C (Heavy)": 80.0, "Grade-D (Specialty)": 95.0}


def _simulate_transition(rng, event_id, from_grade, to_grade, aggressiveness):
    """
    aggressiveness in [0,1]: overall latent "how rushed was this operator" factor.
    Each driver gets its OWN noisy component around that latent factor, so the
    four drivers are correlated with each other (same underlying operator) but
    not identical -- this is what lets the correlation engine find genuinely
    different relationship strengths per variable instead of four copies of
    the same number.
    """
    bw_start = GRADE_TARGET_BW[from_grade]
    bw_target = GRADE_TARGET_BW[to_grade]

    def _component(weight=0.75, noise=0.20):
        c = weight * aggressiveness + rng.uniform(-noise, noise)
        return float(np.clip(c, 0.0, 1.0))

    agg_steam = _component()
    agg_filler = _component()
    agg_speed = _component()
    agg_moist = _component()

    # Ramp rates / step sizes scale with each driver's OWN component
    steam_ramp_rate = 0.15 + agg_steam * 0.9          # units/step
    filler_step_size = 2.0 + agg_filler * 8.0          # abrupt filler flow jump
    speed_delta = 1.0 + agg_speed * 6.0                # machine speed change magnitude
    moisture_lag_steps = int(3 + agg_moist * 10)       # how late moisture loop reacts

    # Weighted blend feeds the actual physical response below (steam/filler/speed
    # dominate overshoot; moisture is handled separately as a lagged disturbance)
    blended_agg = 0.4 * agg_steam + 0.35 * agg_filler + 0.25 * agg_speed

    t = np.arange(N_STEPS)

    # --- Manipulated / supporting variables ---
    stock_flow = 100 + (bw_target - bw_start) * 0.4 + rng.normal(0, 0.8, N_STEPS)
    filler_flow = 20 + filler_step_size * (1 - np.exp(-t / 8)) + rng.normal(0, 0.5, N_STEPS)
    steam_pressure = 50 + steam_ramp_rate * t + rng.normal(0, 0.6, N_STEPS)
    steam_pressure = np.clip(steam_pressure, 40, 120)
    machine_speed = 900 + speed_delta * (1 - np.exp(-t / 6)) + rng.normal(0, 3, N_STEPS)

    # --- Quality variables (what we actually care about) ---
    # Basis weight approaches target with fixed underlying process dynamics
    # (tau is a property of the machine, not the operator). Aggressiveness
    # ONLY affects overshoot magnitude and how long the oscillation persists
    # -- this is what makes "aggressive == worse outcome" actually hold.
    tau = 8.0
    raw_approach = 1 - np.exp(-t / tau)
    overshoot_gain = 1.0 + blended_agg * 3.6
    decay_tau = tau * (1.4 + blended_agg * 3.6)  # aggressive -> oscillation lingers longer
    oscillation = (overshoot_gain * np.exp(-t / decay_tau)
                   * np.sin(t / (tau * 0.9)) * (0.15 + blended_agg) * 3.8)
    # Moisture lag adds its own smaller, independent disturbance to basis weight
    moisture_disturbance = np.zeros(N_STEPS)
    moisture_disturbance[moisture_lag_steps:] += (
        agg_moist * 0.8 * np.exp(-(t[moisture_lag_steps:] - moisture_lag_steps) / 8))
    basis_weight = bw_start + (bw_target - bw_start) * raw_approach + oscillation + moisture_disturbance
    basis_weight += rng.normal(0, 0.15, N_STEPS)

    # Moisture reacts late (lag) -> visible in its own sensor trace too
    moisture = 8.5 + 0.5 * np.sin(t / 15) + rng.normal(0, 0.2, N_STEPS)
    moisture[moisture_lag_steps:] += agg_moist * 1.5 * np.exp(-(t[moisture_lag_steps:] - moisture_lag_steps) / 8)

    ash = 18 + 0.3 * (filler_flow - 20) / 5 + rng.normal(0, 0.3, N_STEPS)
    caliper = 3.2 + 0.02 * (basis_weight - bw_target) + rng.normal(0, 0.03, N_STEPS)

    df = pd.DataFrame({
        "event_id": event_id, "t": t,
        "stock_flow": stock_flow, "filler_flow": filler_flow,
        "steam_pressure": steam_pressure, "machine_speed": machine_speed,
        "moisture": moisture, "ash": ash, "caliper": caliper,
        "basis_weight": basis_weight,
    })

    # --- Outcome metrics ---
    # Off-spec is only meaningful AFTER the nominal ramp window (t >= RAMP_END):
    # during the ramp itself the process is *supposed* to be moving away from
    # old-target, so comparing raw distance-to-final-target the whole time
    # would flag every transition as "off-spec" by construction.
    RAMP_END = 32
    pct_dev_full = 100 * np.abs(basis_weight - bw_target) / bw_target
    pct_dev_post_ramp = pct_dev_full[RAMP_END:]
    max_dev_pct = float(np.max(pct_dev_post_ramp))
    went_offspec = bool(max_dev_pct > SPEC_LIMIT_PCT)

    within_band = pct_dev_full <= SPEC_LIMIT_PCT
    # stabilization time = first index (after ramp window) from which it STAYS within band
    stab_idx = N_STEPS
    for i in range(RAMP_END, N_STEPS):
        if np.all(within_band[i:]):
            stab_idx = i
            break

    meta = {
        "event_id": event_id, "from_grade": from_grade, "to_grade": to_grade,
        "bw_target": bw_target, "aggressiveness": round(aggressiveness, 3),
        "steam_ramp_rate": round(steam_ramp_rate, 3),
        "filler_step_size": round(filler_step_size, 3),
        "speed_delta": round(speed_delta, 3),
        "moisture_lag_steps": moisture_lag_steps,
        "max_deviation_pct": round(max_dev_pct, 3),
        "went_offspec": went_offspec,
        "stabilization_time": stab_idx,
    }
    return df, meta


def generate(n_events=N_EVENTS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    all_ts, all_meta = [], []
    for i in range(n_events):
        from_g, to_g = rng.choice(GRADES, size=2, replace=False)
        # Mix of calm and aggressive operators -> gives us both good & bad cases
        aggressiveness = float(np.clip(rng.beta(2, 3), 0, 1))
        df, meta = _simulate_transition(rng, i, from_g, to_g, aggressiveness)
        all_ts.append(df)
        all_meta.append(meta)
    ts = pd.concat(all_ts, ignore_index=True)
    meta = pd.DataFrame(all_meta)
    return ts, meta


if __name__ == "__main__":
    ts, meta = generate()
    ts.to_csv("events_timeseries.csv", index=False)
    meta.to_csv("events_meta.csv", index=False)
    print(f"Generated {meta['event_id'].nunique()} events, "
          f"{ts.shape[0]} timesteps total.")
    print(f"Off-spec rate: {meta['went_offspec'].mean():.1%}")
    print(meta.head())
