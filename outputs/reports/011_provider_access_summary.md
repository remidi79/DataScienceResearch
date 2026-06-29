# Provider Access Summary

item_type,path,description,credential_required,credential_detected,safe_to_use,notes
code_or_config,/home/platform/DataPlatform/warehouse/ingestion/statsbomb_api_client.py,StatsBomb provider ingestion/client/config artefact,True,False,True,exists
code_or_config,/home/platform/DataPlatform/warehouse/ingestion/statsbomb.py,StatsBomb provider ingestion/client/config artefact,True,False,True,exists
code_or_config,/home/platform/DataPlatform/warehouse/jobs/fetch_statsbomb.py,StatsBomb provider ingestion/client/config artefact,False,False,True,exists
target_config,/home/platform/DataPlatform/config/statsbomb_targets.json,StatsBomb provider ingestion/client/config artefact,False,False,True,exists
target_config,/home/platform/DataPlatform/config/statsbomb_targets_botola.json,StatsBomb provider ingestion/client/config artefact,False,False,True,exists
code_or_config,/home/platform/DataPlatform/airflow/dags/statsbomb_bsg_pipeline.py,StatsBomb provider ingestion/client/config artefact,False,False,True,exists
code_or_config,/home/platform/DataPlatform/airflow/dags/backfill_statsbomb_dag.py,StatsBomb provider ingestion/client/config artefact,False,False,True,exists
code_or_config,/home/platform/DataPlatform/warehouse/jobs/build_statsbomb_provider_stats_marts.py,StatsBomb provider ingestion/client/config artefact,False,False,True,exists
env_config,/home/platform/DataPlatform/.env.example,Contains StatsBomb credential variable names; values not inspected or printed,True,False,True,variable names only; no secret output
environment,process_environment,Runtime StatsBomb credentials,True,False,False,STATSBOMB_API_USERNAME=missing;STATSBOMB_API_PASSWORD=missing;STATSBOMB_API_TOKEN=missing;STATSBOMB_API_BASE=missing

Secrets were not printed.
