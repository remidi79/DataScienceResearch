# Experiment 008 — Full-Population Recalibration & Production-Candidate Bundle

## 1. Objective
Run full-population recalibration and production-candidate gate without deploying or falsely promoting scores.

## 2. Whether full population was available
False. The current root fails the production data-readiness gate.

## 3. Dataset coverage
```json
{
  "competitions": 1,
  "seasons": 1,
  "matches": 11,
  "teams": 18,
  "players": 473,
  "player_season_rows": 489,
  "player_match_rows": 350,
  "events": 35082,
  "lineups": 450
}
```

## 4. Role eligibility recalibration
Role resolution and eligibility summary are written, but full recalibration is blocked by data readiness.

## 5. Metric stability recalibration
Blocked; archived into 008 tables with explicit status.

## 6. Normalization recalibration
Blocked; previous artefacts preserved with blocked status.

## 7. Latent dimension recalibration
Blocked until full population is available.

## 8. Weight recalibration
Blocked; no coefficients changed.

## 9. Full score recalculation
Blocked; no production scores declared.

## 10. Full validation results
Blocked by population readiness; prior validation artefacts are archived into 008 outputs for traceability.

## 11. Production-candidate gate
All roles remain Research Prototype.

## 12. Roles passing / failing
No role passes production-candidate gates on the current root.

## 13. Main blockers by role
See `008_blockers_by_role.csv`.

## 14. Expert review requirements
Expert review remains required before any production promotion.

## 15. Whether config bundle was generated
False.

## 16. Why this is or is not production-ready
Not production-ready: insufficient seasons, competitions, matches, eligible players, and expert review.

## 17. Recommended next step
Load/rebuild the full StatsBomb population and rerun Experiment 008.
