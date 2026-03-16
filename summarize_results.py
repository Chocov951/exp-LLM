import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

def parse_ndcg_for_sort(value: str) -> float:
    text = (value or "").strip()
    if not text or text.upper() == "N/A":
        return -1.0
    try:
        return float(text)
    except ValueError:
        return -1.0


def render_table(records: List[List[str]]) -> str:
    header = ["filter_method", "rerank_method", "filter_size", "bm25_k", "ndcg@10", "time"]
    if not records:
        return "(no rows)"

    widths = [len(h) for h in header]
    for row in records:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt_row(header)]
    lines.append(" | ".join("-" * w for w in widths))
    lines.extend(fmt_row(r) for r in records)
    return "\n".join(lines)


def build_dataset_table(csv_path: Path) -> str:
    if not csv_path.exists():
        return f"(missing file: {csv_path.name})"

    selected_rows: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filter_method = (row.get("filter_method") or "").strip()
            rerank_method = (row.get("rerank_method") or "").strip()
            filter_size = (row.get("filter_size") or "").strip()
            bm25_k = (row.get("bm25_k") or "").strip()
            ndcg10 = (row.get("ndcg@10") or "").strip()

            avg_total_time = (row.get("avg_total_time") or "").strip()
            avg_filter_time = (row.get("avg_filter_time") or "").strip()
            # Prefer total time when present, otherwise fallback to filter time.
            time_value = avg_total_time if avg_total_time and avg_total_time.upper() != "N/A" else avg_filter_time

            selected_rows.append([filter_method, rerank_method, filter_size, bm25_k, ndcg10, time_value])

    selected_rows.sort(key=lambda r: parse_ndcg_for_sort(r[4]), reverse=True)
    return render_table(selected_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize results_trec19.csv and results_trec20.csv.")
    parser.add_argument("folder", type=str, help="Folder containing results_trec19.csv and results_trec20.csv")
    args = parser.parse_args()

    folder = Path(args.folder)
    trec19_table = build_dataset_table(folder / "results_trec19.csv")
    trec20_table = build_dataset_table(folder / "results_trec20.csv")

    print("results_trec19.csv:\n" + trec19_table + "\n")
    print("results_trec20.csv:\n" + trec20_table)


if __name__ == "__main__":
    main()
