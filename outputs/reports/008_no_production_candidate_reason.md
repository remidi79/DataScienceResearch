# Experiment 008 — No Production Candidate Reason

A production-candidate bundle was not generated because the available data root does not meet the minimum production target.

## Dataset summary

- competitions: 1
- seasons: 1
- matches: 11
- teams: 18
- players: 473
- player_season_rows: 489
- player_match_rows: 350
- events: 35082
- lineups: 450

## Failed gates

- at_least_2_seasons: observed 1, minimum 2
- at_least_3_competitions_if_available: observed 1, minimum 3
- enough_matches_for_temporal_validation: observed 11, minimum 100
- enough_teams_for_context_bias_testing: observed 18, minimum 20
- enough_goalkeepers: observed 12, minimum 20
- enough_cb_eligible_players: observed 25, minimum 50
- enough_fb_eligible_players: observed 23, minimum 50
- enough_mid_eligible_players: observed 34, minimum 50
- enough_winger_eligible_players: observed 25, minimum 50
- enough_cf_eligible_players: observed 22, minimum 50

No production coefficients were changed, signed, or deployed.
