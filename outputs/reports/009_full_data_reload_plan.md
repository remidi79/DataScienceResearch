# Experiment 009 — Full Data Reload Plan

Reload DataPlatform in dependency order:

- R001 competition_metadata: source=StatsBomb competitions/seasons endpoints or DataPlatform bronze metadata -> target=metadata/competitions_seasons; validate row counts, required keys, ID consistency, coverage gates.
- R002 silver_matches: source=StatsBomb matches endpoint by competition/season -> target=silver/silver_matches.jsonl; validate row counts, required keys, ID consistency, coverage gates.
- R003 silver_lineups: source=StatsBomb lineups endpoint by match -> target=silver/silver_lineups.jsonl; validate row counts, required keys, ID consistency, coverage gates.
- R004 silver_events: source=StatsBomb events endpoint by match -> target=silver/silver_events.jsonl; validate row counts, required keys, ID consistency, coverage gates.
- R005 player_match_stats_direct: source=StatsBomb player match stats endpoint -> target=marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl; validate row counts, required keys, ID consistency, coverage gates.
- R006 team_match_stats_direct: source=StatsBomb team match stats endpoint -> target=marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl; validate row counts, required keys, ID consistency, coverage gates.
- R007 player_season_stats_direct: source=StatsBomb player season stats endpoint -> target=marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl; validate row counts, required keys, ID consistency, coverage gates.
- R008 team_season_stats_direct: source=StatsBomb team season stats endpoint -> target=marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl; validate row counts, required keys, ID consistency, coverage gates.

The score-engine rerun must not start until all blocking datasets pass the readiness gate.
