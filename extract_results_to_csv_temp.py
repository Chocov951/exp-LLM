import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, NamedTuple, List


# Extract dataset, filter size, filter method, and rank method from filenames like:
# results_trec19_filter10-CustomLLM-qwen72_rank-Listwise-zephyr-W20-S10_top100.json
FILENAME_RE = re.compile(r"results_(trec\d+)_filter(\d+)-(.+?)_rank-(.+?)_top(\d+)\.json$")


def parse_filename(path: Path) -> Optional[Tuple[str, int, str, str, int]]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    dataset, filter_size, filter_method, rank_method, bm25_k = match.groups()
    return dataset, int(filter_size), filter_method, rank_method, int(bm25_k)


DIVISORS = {"trec19": 43, "trec20": 54}


class Metrics(NamedTuple):
    ndcg_5: Optional[float]
    ndcg_10: Optional[float]
    ndcg_20: Optional[float]
    filtered_ndcg_5: Optional[float]
    filtered_ndcg_10: Optional[float]
    filtered_ndcg_20: Optional[float]
    filter_time: Optional[float]
    filter_energy_kwh: Optional[float]
    filter_emissions: Optional[float]
    ranking_time: Optional[float]
    ranking_energy_kwh: Optional[float]
    ranking_emissions: Optional[float]


def as_float(value) -> Optional[float]:
    """Best-effort conversion to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_numeric_field(obj: Dict, keys: List[str]) -> Optional[float]:
    """Extract first numeric value from candidate keys, including nested dict values."""
    if not isinstance(obj, dict):
        return None

    for key in keys:
        if key not in obj:
            continue
        raw = obj.get(key)
        numeric = as_float(raw)
        if numeric is not None:
            return numeric

        if isinstance(raw, dict):
            # Handle nested payloads like {'value': ...} or {'energy_consumed_kwh': ...}.
            for nested_key in (
                "value",
                "duration_s",
                "duration",
                "energy_consumed_kwh",
                "energy_consumed",
                "emissions_kg",
                "emissions",
            ):
                nested_numeric = as_float(raw.get(nested_key))
                if nested_numeric is not None:
                    return nested_numeric
    return None


def load_metrics(path: Path) -> Metrics:
    """Extract all metrics from a results JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    metrics = data.get("metrics", {})
    filtered = metrics.get("filtered", {})
    # Use two_stage_reranking for metrics, fallback to filtered
    two_stage = metrics.get("two_stage_reranking", {})
    if not two_stage:
        two_stage = filtered
    
    codecarbon = data.get("codecarbon", {})
    
    # Extract filter stage metrics
    filter_stage = codecarbon.get("filter", {}) if isinstance(codecarbon, dict) else {}
    filter_time = extract_numeric_field(filter_stage, ["duration_s", "duration"])
    filter_energy_kwh = extract_numeric_field(filter_stage, ["energy_consumed_kwh", "energy_consumed"])
    filter_emissions = extract_numeric_field(filter_stage, ["emissions_kg", "emissions"])
    
    # Extract ranking stage metrics
    ranking_stage = codecarbon.get("ranking", {}) if isinstance(codecarbon, dict) else {}
    ranking_time = extract_numeric_field(ranking_stage, ["duration_s", "duration"])
    ranking_energy_kwh = extract_numeric_field(ranking_stage, ["energy_consumed_kwh", "energy_consumed"])
    ranking_emissions = extract_numeric_field(ranking_stage, ["emissions_kg", "emissions"])
    
    return Metrics(
        ndcg_5=as_float(two_stage.get("ndcg@5")),
        ndcg_10=as_float(two_stage.get("ndcg@10")),
        ndcg_20=as_float(two_stage.get("ndcg@20")),
        filtered_ndcg_5=as_float(filtered.get("ndcg@5")),
        filtered_ndcg_10=as_float(filtered.get("ndcg@10")),
        filtered_ndcg_20=as_float(filtered.get("ndcg@20")),
        filter_time=filter_time,
        filter_energy_kwh=filter_energy_kwh,
        filter_emissions=filter_emissions,
        ranking_time=ranking_time,
        ranking_energy_kwh=ranking_energy_kwh,
        ranking_emissions=ranking_emissions,
    )


def should_skip_filter_method(filter_method: str) -> bool:
    """Skip CustomLLM variants except CustomLLM-vLLM."""
    return (filter_method.startswith("CustomLLM") and not filter_method.startswith("CustomLLM-vLLM"))

def is_customllm_vllm_filter_method(filter_method: str) -> bool:
    return filter_method.startswith("CustomLLM-vLLM-")


def extract_to_csv(folder: Path) -> None:
    """Parse result JSON files and write separate CSV files for each dataset."""
    # Dictionary structure: (dataset, filter_method, rerank_method, filter_size, bm25_k) -> Metrics
    data: Dict[Tuple[str, str, str, int, int], Metrics] = {}
    
    for path in folder.glob("*.json"):
        parsed = parse_filename(path)
        if not parsed:
            continue
        
        dataset, filter_size, filter_method, rank_method, bm25_k = parsed
        if should_skip_filter_method(filter_method):
            continue
        metrics = load_metrics(path)
        key = (dataset, filter_method, rank_method, filter_size, bm25_k)

        if key in data:
            print(
                "Warning: duplicate row key detected "
                f"({dataset}, {filter_method}, {rank_method}, {filter_size}, {bm25_k}) in {path.name}; "
                "overwriting previous value."
            )
        data[key] = metrics
    
    if not data:
        print("No matching JSON files found.")
        return
    
    # Group by dataset and write CSV for each
    by_dataset: Dict[str, List[Tuple[str, str, int, int, Metrics]]] = {}
    for (dataset, filter_method, rank_method, filter_size, bm25_k), metrics in data.items():
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append((filter_method, rank_method, filter_size, bm25_k, metrics))
    
    # Write CSV for each dataset
    for dataset in sorted(by_dataset.keys()):
        write_csv_for_dataset(folder, dataset, by_dataset[dataset])


def write_csv_for_dataset(folder: Path, dataset: str, rows_data: List[Tuple[str, str, int, int, Metrics]]) -> None:
    """Write CSV file for a specific dataset."""
    divisor = DIVISORS.get(dataset, 1)
    output_file = folder / f"results_{dataset}.csv"
    
    fieldnames = [
        "filter_method",
        "rerank_method", 
        "filter_size",
        "bm25_k",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "avg_total_time",
        "avg_total_consumption_wh",
        "avg_total_emissions_g",
        "avg_filter_time",
        "avg_filter_consumption_wh",
        "avg_filter_emissions_g",
        "avg_ranking_time",
        "avg_ranking_consumption_wh",
        "avg_ranking_emissions_g",
    ]

    def sum_if_both(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None and b is None: return None
        if a is None : return b
        if b is None : return a
        return a + b
    
    filtered_rows_data: List[Tuple[str, str, int, int, Metrics]] = []
    for filter_method, rank_method, filter_size, bm25_k, metrics in rows_data:
        filtered_rows_data.append((filter_method, rank_method, filter_size, bm25_k, metrics))

    # Prepare row data with averages calculated
    rows = []
    no_rerank_candidates: Dict[Tuple[str, int, int], Metrics] = {}
    for filter_method, rank_method, filter_size, bm25_k, metrics in filtered_rows_data:
        if is_customllm_vllm_filter_method(filter_method):
            no_rerank_key = (filter_method, filter_size, bm25_k)
            if no_rerank_key not in no_rerank_candidates:
                no_rerank_candidates[no_rerank_key] = metrics

        avg_ndcg_5 = metrics.ndcg_5
        avg_ndcg_10 = metrics.ndcg_10
        avg_ndcg_20 = metrics.ndcg_20
        avg_filter_time = metrics.filter_time
        avg_filter_energy = metrics.filter_energy_kwh
        avg_filter_emissions = metrics.filter_emissions
        avg_ranking_time = metrics.ranking_time
        avg_ranking_energy = metrics.ranking_energy_kwh
        avg_ranking_emissions = metrics.ranking_emissions
        
        # Divide times and energy by divisor to get per-query averages
        if avg_filter_time is not None:
            avg_filter_time = avg_filter_time / divisor
        if avg_filter_energy is not None:
            avg_filter_energy = avg_filter_energy / divisor
        if avg_filter_emissions is not None:
            avg_filter_emissions = avg_filter_emissions / divisor
        if avg_ranking_time is not None:
            avg_ranking_time = avg_ranking_time / divisor
        if avg_ranking_energy is not None:
            avg_ranking_energy = avg_ranking_energy / divisor
        if avg_ranking_emissions is not None:
            avg_ranking_emissions = avg_ranking_emissions / divisor

        # Additional scaling requested by user.
        if avg_filter_energy is not None:
            avg_filter_energy = avg_filter_energy * 1000.0
        if avg_filter_emissions is not None:
            avg_filter_emissions = avg_filter_emissions * 1000.0
        if avg_ranking_energy is not None:
            avg_ranking_energy = avg_ranking_energy * 1000.0
        if avg_ranking_emissions is not None:
            avg_ranking_emissions = avg_ranking_emissions * 1000.0

        avg_total_time = sum_if_both(avg_filter_time, avg_ranking_time)
        avg_total_consumption = sum_if_both(avg_filter_energy, avg_ranking_energy)
        avg_total_emissions = sum_if_both(avg_filter_emissions, avg_ranking_emissions)
        
        rows.append({
            "filter_method": filter_method,
            "rerank_method": rank_method,
            "filter_size": filter_size,
            "bm25_k": bm25_k,
            "ndcg@5": format_metric(avg_ndcg_5),
            "ndcg@10": format_metric(avg_ndcg_10),
            "ndcg@20": format_metric(avg_ndcg_20),
            "avg_total_time": format_metric(avg_total_time),
            "avg_total_consumption_wh": format_metric(avg_total_consumption),
            "avg_total_emissions_g": format_metric(avg_total_emissions),
            "avg_filter_time": format_metric(avg_filter_time),
            "avg_filter_consumption_wh": format_metric(avg_filter_energy),
            "avg_filter_emissions_g": format_metric(avg_filter_emissions),
            "avg_ranking_time": format_metric(avg_ranking_time),
            "avg_ranking_consumption_wh": format_metric(avg_ranking_energy),
            "avg_ranking_emissions_g": format_metric(avg_ranking_emissions),
        })

    # Add synthetic No-Rerank rows for CustomLLM-vLLM-* using filtered metrics.
    existing_keys = {
        (str(r["filter_method"]), str(r["rerank_method"]), int(r["filter_size"]), int(r["bm25_k"])) for r in rows
    }
    for (filter_method, filter_size, bm25_k), metrics in sorted(
        no_rerank_candidates.items(), key=lambda x: (x[0][0].lower(), x[0][1], x[0][2])
    ):
        key = (filter_method, "No-Rerank", filter_size, bm25_k)
        if key in existing_keys:
            continue

        no_rerank_filter_time = metrics.filter_time
        no_rerank_filter_energy = metrics.filter_energy_kwh
        no_rerank_filter_emissions = metrics.filter_emissions

        if no_rerank_filter_time is not None:
            no_rerank_filter_time = no_rerank_filter_time / divisor
        if no_rerank_filter_energy is not None:
            no_rerank_filter_energy = (no_rerank_filter_energy / divisor) * 1000.0
        if no_rerank_filter_emissions is not None:
            no_rerank_filter_emissions = (no_rerank_filter_emissions / divisor) * 1000.0

        rows.append(
            {
                "filter_method": filter_method,
                "rerank_method": "No-Rerank",
                "filter_size": filter_size,
                "bm25_k": bm25_k,
                "ndcg@5": format_metric(metrics.filtered_ndcg_5),
                "ndcg@10": format_metric(metrics.filtered_ndcg_10),
                "ndcg@20": format_metric(metrics.filtered_ndcg_20),
                "avg_total_time": "",
                "avg_total_consumption_wh": "",
                "avg_total_emissions_g": "",
                "avg_filter_time": format_metric(no_rerank_filter_time),
                "avg_filter_consumption_wh": format_metric(no_rerank_filter_energy),
                "avg_filter_emissions_g": format_metric(no_rerank_filter_emissions),
                "avg_ranking_time": "",
                "avg_ranking_consumption_wh": "",
                "avg_ranking_emissions_g": "",
            }
        )
        existing_keys.add(key)

    # Sort by filter method, rerank method, then filter size
    rows.sort(
        key=lambda r: (
            r["filter_method"].lower(),
            r["rerank_method"].lower(),
            int(r["filter_size"]),
            int(r["bm25_k"]),
        )
    )
    
    # Write CSV
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ {output_file.name} ({len(rows)} rows)")


def format_metric(value: Optional[float]) -> str:
    """Format a metric value for CSV output."""
    if value is None:
        return ""
    # Use appropriate precision based on magnitude
    if abs(value) < 0.1:
        return f"{value:.6f}"
    elif abs(value) < 10:
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract results from JSON files and write to CSV files (one per dataset)."
    )
    parser.add_argument("folder", type=str, help="Folder containing results_*.json files")
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory.")
        return
    
    print(f"Processing results from {folder}...")
    extract_to_csv(folder)
    print("Done!")


if __name__ == "__main__":
    main()