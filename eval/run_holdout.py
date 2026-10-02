from __future__ import annotations

import argparse
import json

from .run_benchmark import run
from .validate_holdout import load_holdout_records, validate_holdout_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    parser.add_argument("--mode", choices=["routing", "full"], default="routing")
    args = parser.parse_args()

    records = load_holdout_records()
    validate_holdout_records(records)
    output = run(
        args.version,
        args.mode,
        records=records,
        result_label="holdout",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
