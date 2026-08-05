from __future__ import annotations

import argparse

from pipeline_utils import SUMMARY_CSV, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment CSV records.")
    parser.add_argument("--project", help="Filter by project path.")
    parser.add_argument("--name", help="Filter by run name prefix.")
    args = parser.parse_args()

    summary = summarize_records(project=args.project, name_prefix=args.name)
    if summary.empty:
        print("No experiment records found for the requested filter.")
        return

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print(f"\nsummary_csv={SUMMARY_CSV}")


if __name__ == "__main__":
    main()
