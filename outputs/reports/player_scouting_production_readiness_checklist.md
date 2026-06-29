# Player Scouting Production Readiness Checklist

Production status: NOT READY.

A production candidate may only proceed when every item below is complete.

- [ ] StatsBomb credentials active.
- [ ] Provider access verified with redacted preflight.
- [ ] Full target warehouse loaded.
- [ ] Experiment 012 licensed backfill executed successfully.
- [ ] Experiment 011 validation passes.
- [ ] Experiment 010 validation passes.
- [ ] Experiment 009 validation passes.
- [ ] Full research pipeline rerun on full warehouse.
- [ ] Experiment 008 production-candidate gate passes.
- [ ] Expert review completed.
- [ ] Production bundle generated only after gates pass.
- [ ] API integration only after production candidate bundle exists.

Current known blockers:

- StatsBomb credentials/provider access are missing or blocked.
- Full target warehouse remains incomplete.
- Licensed backfill has not executed.
- Current local sample is only 11 matches, 1 competition, and 1 season.
- Experiment 015 is research-only, event-derived only, not provider-direct, and not production-ready.
