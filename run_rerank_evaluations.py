#!/usr/bin/env python3
"""
Merged evaluation runner.

Iterates over all (dataset, bm25_topk, filter_topk, model, filter_method)
combinations while loading the dataset and BM25 cache only once per
(dataset, bm25_topk) pair — avoiding the repeated subprocess overhead of
calling `python evaluate_and_save.py …` in a loop.

Mirrors exactly the stage=rerank logic of evaluate_and_save.py::main().
"""

import os
import json
import time
import csv
from typing import Dict, Tuple

from ranx import Qrels, Run, evaluate
from utils import load_dataset


# ========================================
# Parameter grid (edit these as needed)
# ========================================

DATASETS       = ["trec20", "trec19"]
MODEL_NAMES    = ["qwen4", "smollm", "gpt", "qwen30", "zephyr", "vicuna"]
BM25_TOPKS     = [100]
REJECT_NUMBERS = [10, 20, 30]
FILTER_METHODS = ["Pointwise", "BERT-all-MiniLM-L12-v2", "BERT-bge-m3"]

OUTPUT_DIR            = "rankllm_results"
LISTWISE_WINDOW_SIZE  = 'full'
LISTWISE_STRIDE       = 10


# ========================================
# Helper functions (from evaluate_and_save.py)
# ========================================

def load_bm25_results(dataset: str, bm25_topk: int) -> Tuple[Dict, Dict]:
    """Load BM25 rundict and passage cache from disk (read-only, no recomputation)."""
    cache_file  = f'rundicts/rundict_{dataset}_bm25.json'
    queries_file = f'rundicts/queries_test_{dataset}_bm25.json'

    if os.path.exists(cache_file) and os.path.exists(queries_file):
        print(f"  Loading cached BM25 results from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            bm25_rundict = json.load(f)
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_with_passages = json.load(f)
        return bm25_rundict, queries_with_passages

    print(f"  Warning: BM25 cache not found for {dataset}, returning empty results")
    return {}, {}


def evaluate_metrics(qrels: Dict, run: Dict) -> Dict:
    metrics = [
        'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@20',
        'recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50',
        'mrr@10', 'mrr@20',
        'precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50',
    ]
    try:
        return evaluate(Qrels(qrels), Run(run), metrics, make_comparable=True)
    except Exception as e:
        print(f"  Error evaluating metrics: {e}")
        return {}


def save_results(
    dataset: str, bm25_topk: int, filter_topk: int,
    filter_method_name: str, ranking_method_name: str,
    metrics_dict: Dict, codecarbon_metrics: Dict,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    filename = (
        f"results_{dataset}_"
        f"filter{filter_topk}-{filter_method_name}_"
        f"rank-{ranking_method_name}_"
        f"top{bm25_topk}.json"
    )
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"  Warning: {filepath} already exists — skipping.")
        return False  # skipped

    full_results = {
        'config': {
            'dataset': dataset,
            'filter_method': filter_method_name,
            'ranking_method': ranking_method_name,
            'stage': 'rerank',
            'bm25_topk': bm25_topk,
            'filter_topk': filter_topk,
        },
        'metrics': metrics_dict,
        'codecarbon': codecarbon_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=4)

    print(f"  Results saved to: {filepath}")
    return True  # saved


# ========================================
# Per-combination evaluation
# ========================================

def run_evaluation(
    dataset: str,
    bm: int,
    rj: int,
    model_name: str,
    filter_method: str,
    qrels: Dict,
    bm25_rundict: Dict,
    output_dir: str = "rankllm_results",
):
    """
    Evaluate one (dataset, bm25_topk, filter_topk, model, filter_method) combination.

    Reproduces the behaviour of:
        python evaluate_and_save.py
            --dataset {dataset}
            --bm25_topk {bm}
            --filter_topk {rj}
            --load_filtered_rundict "rankllm_results/rundicts/filtered_{dataset}_{filter_method}_filter{rej}_top{bm}.json"
            --listwise_model custom_llm
            --custom_model {model_name}
            --ranking_method listwise
            --stage rerank
    """
    # Reproduce rej logic from the original generate script
    # (rej only affected the --load_filtered_rundict path whose only use was
    #  extracting filter_method_name via split("_")[2], i.e. == filter_method)
    rej = rj if filter_method == "CustomLLM-qwen30" else 100  # noqa: F841 (kept for clarity)

    # Names derived exactly as in evaluate_and_save.py for stage=rerank
    filter_method_name  = filter_method                                                  # split("_")[2] of load_filtered_rundict basename
    filter_number       = f"filter{rj}"                                                  # from --filter_topk
    ranking_method_name = f"Listwise-{model_name}-W{LISTWISE_WINDOW_SIZE}-S{LISTWISE_STRIDE}"

    filtered_output_path = os.path.join(
        output_dir, "rundicts",
        f"filtered_{dataset}_{filter_method_name}_filter100_top{bm}.json",
    )
    ranking_output_path = os.path.join(
        output_dir, "rundicts",
        f"ranked_{dataset}_{ranking_method_name}_{filter_number}_top{bm}.json",
    )

    print(f"\n  running results_{dataset}_filter{rej}_{filter_method}_rank{model_name}")

    try:
        with open(ranking_output_path, 'r', encoding='utf-8') as f:
            rundict_rerank = json.load(f)
    except FileNotFoundError:
        print(f"  Error: Reranked results not found at {ranking_output_path}")
        return "not_found"

    try:
        with open(filtered_output_path, 'r', encoding='utf-8') as f:
            filtered_rundict = json.load(f)
    except FileNotFoundError:
        print(f"  Error: Filtered rundict not found at {filtered_output_path}")
        return "not_found"

    # --- Evaluate ---
    bm25_metrics     = evaluate_metrics(qrels, bm25_rundict) if bm25_rundict else {}
    filtered_metrics = evaluate_metrics(qrels, filtered_rundict)
    rerank_metrics   = evaluate_metrics(qrels, rundict_rerank)

    # print("  Two-stage reranking metrics:")
    # print(json.dumps(rerank_metrics, indent=4))

    # --- CodeCarbon (read pre-existing CSV if present) ---
    codecarbon_metrics = {}
    emissions_dir      = os.path.join(output_dir, 'emissions')
    filter_csv_path    = os.path.join(
        emissions_dir,
        f"emissions_filter_{dataset}_{filter_method_name}_filter100_top{bm}.csv",
    )
    if os.path.exists(filter_csv_path):
        with open(filter_csv_path, 'r') as f:
            reader_list = list(csv.DictReader(f))
        if reader_list:
            d = reader_list[-1]
            codecarbon_metrics['filter'] = {
                'emissions_kg':       float(d.get('emissions',      0)),
                'energy_consumed_kwh': float(d.get('energy_consumed', 0)),
                'duration_s':          float(d.get('duration',        0)),
            }
    
    rank_csv_path      = os.path.join(
        emissions_dir,
        f"emissions_ranking_{dataset}_{ranking_method_name}_{filter_number}_top{bm}.csv",
    )
    if os.path.exists(rank_csv_path):
        with open(rank_csv_path, 'r') as f:
            reader_list = list(csv.DictReader(f))
        if reader_list:
            d = reader_list[-1]
            codecarbon_metrics['ranking'] = {
                'emissions_kg':       float(d.get('emissions',      0)),
                'energy_consumed_kwh': float(d.get('energy_consumed', 0)),
                'duration_s':          float(d.get('duration',        0)),
            }

    all_metrics = {
        'bm25':                bm25_metrics,
        'filtered':            filtered_metrics,
        'two_stage_reranking': rerank_metrics,
    }

    written = save_results(
        dataset=dataset, bm25_topk=bm, filter_topk=rj,
        filter_method_name=filter_method_name,
        ranking_method_name=ranking_method_name,
        metrics_dict=all_metrics, codecarbon_metrics=codecarbon_metrics,
        output_dir=output_dir,
    )
    return "saved" if written else "skipped"


# ========================================
# Main loop
# ========================================

if __name__ == "__main__":
    from collections import defaultdict

    # counters[dim_key] = {"saved": int, "skipped": int, "not_found": int}
    counts_model         = defaultdict(lambda: {"saved": 0, "skipped": 0, "not_found": 0})
    counts_rj            = defaultdict(lambda: {"saved": 0, "skipped": 0, "not_found": 0})
    counts_filter_method = defaultdict(lambda: {"saved": 0, "skipped": 0, "not_found": 0})
    total                = {"saved": 0, "skipped": 0, "not_found": 0}

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # Load dataset ONCE per dataset
        print("Loading qrels …")
        _, _, qrels = load_dataset(dataset)

        for bm in BM25_TOPKS:
            # Load BM25 cache ONCE per (dataset, bm25_topk)
            print(f"Loading BM25 top-{bm} results …")
            bm25_rundict, _ = load_bm25_results(dataset, bm)

            for rj in REJECT_NUMBERS:
                for model_name in MODEL_NAMES:
                    for filter_method in FILTER_METHODS:
                        status = run_evaluation(
                            dataset=dataset,
                            bm=bm,
                            rj=rj,
                            model_name=model_name,
                            filter_method=filter_method,
                            qrels=qrels,
                            bm25_rundict=bm25_rundict,
                            output_dir=OUTPUT_DIR,
                        )
                        counts_model[model_name][status]         += 1
                        counts_rj[rj][status]                    += 1
                        counts_filter_method[filter_method][status] += 1
                        total[status]                            += 1

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    def _fmt(d: dict) -> str:
        n = sum(d.values())
        return (f"saved {d['saved']}/{n} "
                f"| skipped {d['skipped']}/{n} "
                f"| not_found {d['not_found']}/{n}")

    sep = "="*60
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)

    print(f"\nTotal  : {_fmt(total)}")

    print("\nBy model:")
    for key in MODEL_NAMES:
        print(f"  {key:<20} {_fmt(counts_model[key])}")

    print("\nBy reject number (filter_topk):")
    for key in REJECT_NUMBERS:
        print(f"  rj={key:<5}              {_fmt(counts_rj[key])}")

    print("\nBy filter method:")
    for key in FILTER_METHODS:
        print(f"  {key:<30} {_fmt(counts_filter_method[key])}")

    print(sep)
