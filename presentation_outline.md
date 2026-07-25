# Presentation Outline — Grade Change Intelligence

I don't have access to the actual template file (it's behind a link on your
HirePro portal I can't reach). Paste this content into it — the structure
below should map onto whatever standard sections it has (problem, approach,
architecture, results, limitations).

---

**Slide 1 — Problem**
Grade changes are high-loss events: mills produce off-spec paper while
quality variables stabilize after a setpoint change. Goal: predict the risk
before the limit is breached, and recommend setpoints to reduce it.

**Slide 2 — Approach**
Interpretable statistics over a black-box model, on purpose: correlation
analysis + percentile envelopes + linear trajectory extrapolation. Every
output can be traced to a specific number, which is what makes the
"rationale" and "source of inference" requirements possible to satisfy
honestly.

**Slide 3 — Architecture**
Three-stage pipeline: synthetic data generator → analysis engine (pure
functions) → Streamlit dashboard. (Use the architecture diagram from
README.md.)

**Slide 4 — Data**
State plainly that no real historian data was available, so the dataset is
synthetic — 140 simulated grade-change events with realistic causal
structure (ramp rate, filler step size, speed delta, moisture lag each
independently affect overshoot and stabilization time). Off-spec rate ~20%,
by design, to have both success and failure cases.

**Slide 5 — Correlations found**
Show the correlation table from Tab 2. Steam ramp rate has the strongest
relationship to deviation (r≈0.80); moisture lag the weakest of the four
(r≈0.59) — still worth flagging since it's not currently used in the
existing QCS system.

**Slide 6 — Live risk prediction**
Screenshot of Tab 1: basis weight trace, spec band, projected trajectory,
risk gauge (green/yellow/red).

**Slide 7 — Recommendations + rationale**
Screenshot of a recommendation card showing text, rationale, source tag,
and confidence level.

**Slide 8 — Accept/reject loop**
Screenshot of Tab 3. Explain this is the mechanism for evaluating
recommendation quality over time (deliverable #6).

**Slide 9 — Honest limitations**
Pull directly from the "Honest limitations" section of README.md. Don't
skip this slide — a judge is more convinced by a team that knows its gaps
than one that claims a perfect solution.

**Slide 10 — What a production version would need**
Real historian data to re-validate correlations; a nonlinear trajectory
model; a feedback loop that ties accepted recommendations back to measured
outcome, not just acceptance rate.
