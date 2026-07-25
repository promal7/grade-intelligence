# Grade Change Intelligence

**Live demo:** https://grade-intelligence-gfspnpjwn7fn68acow5aen.streamlit.app/

## The problem, in short

When a paper machine switches grades, the quality variables (basis weight
especially) drift while the process re-stabilizes. If that drift exceeds
spec before things settle, you get broke, off-spec product, or wasted time
waiting it out. Experienced operators develop a feel for which levers cause
the most trouble — this project tries to make that intuition explicit and
queryable instead of living only in someone's head. Concretely: predict when
a transition is trending off-spec before it happens, explain why using real
correlations from historical data, and recommend setpoints that actually fix
it.

## How it's put together

Three pieces, each doing one job:

**`data_generator.py`** builds the synthetic dataset. Every event has an
underlying "how aggressively was this transition tuned" factor that drives
both the process variables (steam ramp rate, filler step size, etc.) and the
outcome (overshoot, stabilization time). That's what gives the correlation
analysis something real to find instead of noise.

**`analysis_engine.py`** is the actual brain — four plain functions, no
model training involved:
- `find_correlations()` — checks all seven named variables (stock flow,
  filler flow, steam pressure, machine speed, moisture, ash, caliper)
  against deviation and stabilization time
- `assess_trajectory_risk()` — fits a trend to the last few readings of a
  live transition and flags risk before the spec limit is actually crossed
- `estimate_stabilization_impact()` — turns "this correlates with
  stabilization time" into an actual number, e.g. "expect to save about 4
  steps"
- `generate_recommendations()` — ties it together into setpoint suggestions,
  each one tagged with where it came from: a hard recipe limit, a
  historical correlation, or a live trajectory projection

I went with plain statistics here on purpose rather than a trained model.
It's slower to sound impressive but every recommendation can point at an
exact number and say why — which matters a lot more than accuracy you can't
explain when someone asks "why did it suggest that."

**`app.py`** is the Streamlit dashboard. Pick a historical event, scrub a
time slider to simulate watching it live, and the recommendations panel
updates in real time. There's also a full correlation explorer and an
accept/reject log so recommendation quality can be tracked over time instead
of just trusted blindly.

## Running it yourself

```bash
pip install -r requirements.txt
python data_generator.py     # regenerates the two CSVs if you want fresh data
streamlit run app.py
```

## Rubric mapping, if you're grading this quickly

| Ask | Where |
|---|---|
| Predict off-spec risk before the limit is hit | `assess_trajectory_risk()`, Live Monitor tab |
| Recommend setpoints to stay in safe limits | `generate_recommendations()` |
| Reduce stabilization time | `estimate_stabilization_impact()` — real numbers attached to each recommendation |
| Rationale for every prediction/recommendation | `rationale` field on every card |
| Correlations across recipe + historical data | `find_correlations()` (all 7 named variables) + `RECIPE_LIMITS` |
| Tag source of every suggestion | `source_of_inference`: `recipe`, `historical_correlation`, or `trajectory_extrapolation` |
| Accept/reject, recorded for evaluation | Recommendation Log tab |
