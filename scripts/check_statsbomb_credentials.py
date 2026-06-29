#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

METHODS = {
    "username_password": ["STATSBOMB_USERNAME", "STATSBOMB_PASSWORD"],
    "api_token": ["STATSBOMB_API_TOKEN"],
    "client_credentials": ["STATSBOMB_CLIENT_ID", "STATSBOMB_CLIENT_SECRET"],
    "auth_token": ["STATSBOMB_AUTH_TOKEN"],
}
ALIASES = {
    "STATSBOMB_USERNAME": ["STATSBOMB_USERNAME", "STATSBOMB_API_USERNAME"],
    "STATSBOMB_PASSWORD": ["STATSBOMB_PASSWORD", "STATSBOMB_API_PASSWORD"],
}


def _present(name: str) -> bool:
    candidates = ALIASES.get(name, [name])
    return any(bool(os.environ.get(candidate)) for candidate in candidates)


def check() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, variables in METHODS.items():
        missing = [var for var in variables if not _present(var)]
        rows.append(
            {
                "credential_method": method,
                "detected": not missing,
                "missing_variables": missing,
                "safe_to_attempt_provider_access": not missing,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check StatsBomb credential presence without printing values.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()
    rows = check()
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            status = "detected" if row["detected"] else "missing"
            missing_vars = row["missing_variables"]
            missing = ",".join(missing_vars) if isinstance(missing_vars, list) else ""
            print(f"{row['credential_method']}: {status}; missing={missing}; safe_to_attempt_provider_access={row['safe_to_attempt_provider_access']}")
    return 0 if any(bool(row["safe_to_attempt_provider_access"]) for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
