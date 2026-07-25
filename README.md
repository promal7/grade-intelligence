# Grade Change Intelligence

A dashboard that predicts when a paper-machine grade change is at risk of
going off-spec, explains *why* using correlations found in historical data,
and recommends setpoints — with every recommendation tagged by its source of
inference and logged when accepted or rejected.

## Why this approach, not a trained ML model

Given the time available, training and validating a real predictive model
(train/test split, hyperparameter tuning, defending its internals) wasn't
realistic — and more importantly, it isn't what the rubric is actually
asking for. Deliverable #4 explicitly asks for the *rationale* behind every
prediction and recommendation. A black-box model can't give you that as
directly as an interpretable statistic can. So the engine is built entirely
on methods you can point at and explain in one sentence:

- **Pearson/Spearman correlation** — which loop parameters actually relate
  to deviation and stabilization time, and how strongly.
- **Percentile envelopes** — what values *successful* historical transitions
  stayed under, used as the "safe zone" for recommendations.
- **Linear trajectory extrapolation** — fit a trend to the last few points
  of the live basis-weight trace, project forward, flag risk before the
  spec limit is actually breached.

## Architecture / building blocks

```
data_generator.py                analysis_engine.py                 app.py (Streamlit)
------------------                ------------------                 -------------------
Simulates 140 historical    -->   find_correlations()          -->   Tab 1: Live Monitor
grade-change transitions          safe_envelope()                    - risk gauge + chart
(events_meta.csv +                assess_trajectory_risk()           - recommendations w/
events_timeseries.csv)            generate_recommendations()           accept/reject buttons
                                                                  -->   Tab 2: Correlations
                                                                        - correlation table
                                                                        - safe envelope table
                                                                        - impact ranking chart
                                                                  -->   Tab 3: Recommendation Log
                                                                        - accept/reject history
                                                                        - acceptance rate metric
                                                                        (writes to
                                                                        recommendation_log.csv)
```

Communication between modules is file-based and function-based, not a live
service: `data_generator.py` writes two CSVs once; `analysis_engine.py` is a
pure-function library with no side effects (easy to unit test, easy to
swap out later for a real historian connection); `app.py` is the only
stateful piece, and its only side effect is appending to
`recommendation_log.csv`.

## Module explanations

**`data_generator.py`** — Produces synthetic data standing in for real
Honeywell QCS/DCS historian data, which we don't have access to. Each event
simulates one grade-change transition with realistic causal structure
deliberately built in: steam pressure ramp rate, filler flow step size,
machine speed delta, and moisture-loop lag each get their own semi-independent
"how aggressively was this tuned" component, and that component drives both
the process variable itself *and* the resulting overshoot/stabilization
outcome. This is what gives the correlation engine real signal to find
instead of noise. Off-spec is evaluated only after a physically-grounded
ramp window (the process needs time to travel from old target to new target
regardless of tuning quality — see code comments for the derivation).

**`analysis_engine.py`** — Three pure functions, one per graded capability:
`find_correlations()` (deliverable #3: new correlations), `assess_trajectory_risk()`
(deliverable #1: predict before spec is exceeded), `generate_recommendations()`
(deliverable #2 + #5: setpoint recommendations with source-of-inference tags).

**`app.py`** — Streamlit dashboard. The sidebar lets you pick a historical
event and scrub a time slider to simulate watching it live. Tab 1 shows the
live trend, projected trajectory, and recommendations. Tab 2 shows the
correlation/impact analysis. Tab 3 is the accept/reject audit log.

## Running it

```bash
pip install -r requirements.txt
python data_generator.py     # regenerates events_meta.csv / events_timeseries.csv
streamlit run app.py
```

## Honest limitations (documented, not hidden)

- **Synthetic data, not real historian data.** The causal relationships were
  designed to be realistic and are explicit in the code, but they are not
  validated against real Honeywell QCS output. A real deployment would need
  to re-derive `safe_envelope()` and `find_correlations()` against actual
  historian data before trusting any recommendation.
- **Trajectory extrapolation is linear.** It's deliberately simple so it's
  explainable, but it will under-predict risk for transitions with strong
  nonlinear (e.g. oscillatory) dynamics beyond the fit window. A production
  version should widen this to a damped-oscillation fit.
- **Recommendation confidence is derived from correlation strength/p-value,
  not from validating the recommendation's actual downstream effect** — the
  accept/reject log in Tab 3 is the mechanism intended to close that loop
  over time, but with only synthetic data there's no ground truth yet on
  whether accepted recommendations actually reduced deviation.
- **No cross-grade generalization check.** Correlations are pooled across
  all grade pairs; a rarer grade pair with too few historical events would
  get a less reliable envelope than a common one, and the current code
  doesn't flag that.

## Mapping to the challenge deliverables

| Deliverable | Where |
|---|---|
| Predict off-spec risk before limit exceeded | `assess_trajectory_risk()`, Tab 1 risk gauge |
| Recommend setpoints for safe operating limits | `generate_recommendations()`, Tab 1 |
| Reduce stabilization time | Tab 2 "High-impact parameters on stabilization time" |
| Rationale behind every prediction/recommendation | `rationale` field on every recommendation |
| Use recipe/historical data, find new correlations | `find_correlations()`, Tab 2 |
| Tag every suggestion with source of inference | `source_of_inference` field (`historical_correlation` / `trajectory_extrapolation`) |
| Accept/reject loop, recorded for evaluation | Tab 3, `recommendation_log.csv` |
