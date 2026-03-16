#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
from beir.datasets.data_loader import GenericDataLoader
from ranx import Qrels, Run, evaluate
import csv
from utils import load_dataset

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RankLLM evaluation pipeline with two-stage strategy")
    
    # Dataset and model configuration
    parser.add_argument("--dataset", type=str, default="scifact", 
                       help="Dataset to test (scifact, trec-covid, fiqa, etc.)")
    parser.add_argument("--custom_model", type=str, default=None,
                       help="Model to use for custom LLM ranking")
    parser.add_argument("--bm25_topk", type=int, default=100,
                       help="Top-k documents from BM25 to rerank")
    
    # Two-stage strategy configuration
    parser.add_argument("--filter_method", type=str, default="custom_llm",
                       choices=["custom_llm", "custom_llm_vllm", "pointwise", "bert_index", "RankLLM_custom"],
                       help="Method for filter stage")
    parser.add_argument("--filter_topk", type=int, default=10,
                       help="Number of passages to keep after filter stage")
    parser.add_argument("--ranking_method", type=str, default="custom_llm",
                       choices=["custom_llm", "custom_llm_vllm", "listwise"],
                       help="Method for ranking stage")
    parser.add_argument("--stage", type=str, default="both", # If only rerank is selected and no filtered rundict is provided, will single-pass rerank BM25 results
                       choices=["filter", "rerank", "both"],
                       help="Select which stage(s) to run: filter only, rerank only, or both")
    parser.add_argument("--load_filtered_rundict", type=str, default=None,
                       help="Path to precomputed filtered rundict JSON; if None, rerank directly from BM25")
    
    # RankLLM specific options
    parser.add_argument("--listwise_model", type=str, default="zephyr",
                       choices=["zephyr", "vicuna", "custom_llm"],
                       help="RankLLM model name if using rankllm methods")
    parser.add_argument("--listwise_window_size", type=lambda x: x if x == "full" else int(x), default=20,
                        help="Window size for RankLLM listwise reranker")
    parser.add_argument("--listwise_stride", type=int, default=10,
                        help="Stride for RankLLM listwise reranker")
    parser.add_argument("--bert_model", type=str, default="bge-m3",
                       help="BERT model for BERT index filtering")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="rankllm_results",
                       help="Directory to save results")
    parser.add_argument("--vllm_seed", type=int, default=0,
                       help="Seed for deterministic vLLM generation")
    
    return parser.parse_args()
        

# ========================================
# Main Pipeline Functions
# ========================================


def load_bm25_results(dataset: str, bm25_topk: int) -> Tuple[Dict, Dict]:
    """
    Load or compute BM25 results using pyserini.
    
    Returns:
        bm25_rundict: Dict of qid -> {doc_id: score}
        queries_with_passages: Dict of qid -> {query: str, passages: {doc_id: text}}
    """
    cache_file = f'rundicts/rundict_{dataset}_bm25.json'
    queries_file = f'rundicts/queries_test_{dataset}_bm25.json'
    
    if os.path.exists(cache_file) and os.path.exists(queries_file):
        print(f"Loading cached BM25 results from {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            bm25_rundict = json.load(f)
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_with_passages = json.load(f)
        return bm25_rundict, queries_with_passages
    
    print(f"Computing BM25 results for {dataset}...")
    
    # Try to load dataset with BEIR
    try:
        corpus, queries, qrels = GenericDataLoader(f"datasets/{dataset}").load(split="test")
        corpus = {k: v['text'] if isinstance(v, dict) else v for k, v in corpus.items()}
    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return {}, {}
    
    # For now, return empty results as pyserini index might not be available
    # In practice, would use LuceneSearcher or create BM25 index
    print("Warning: BM25 index not available, returning empty results")
    print("To use this script, ensure pyserini indexes are available or pre-compute BM25 results")
    
    bm25_rundict = {}
    queries_with_passages = {}
    
    # Create directory if it doesn't exist
    os.makedirs('rundicts', exist_ok=True)
    
    return bm25_rundict, queries_with_passages


def evaluate_metrics(qrels: Dict, run: Dict) -> Dict:
    """Evaluate ranking metrics."""
    metrics = [
        'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@20',
        'recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50',
        'mrr@10', 'mrr@20',
        'precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50'
    ]
    
    try:
        results = evaluate(Qrels(qrels), Run(run), metrics, make_comparable=True)
        return results
    except Exception as e:
        print(f"Error evaluating metrics: {e}")
        return {}


def save_results(args, metrics_dict: Dict, codecarbon_metrics: Dict, output_dir: str, filter_method_name: str, ranking_method_name: str) -> str:
    """Save all metrics to JSON file with descriptive filename."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with hyperparameters
    filename = (
        f"results_"
        f"{args.dataset}_"
        f"filter{args.filter_topk}-{filter_method_name}_"
        f"rank-{ranking_method_name}_"
        f"top{args.bm25_topk}.json"
    )
    
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"Warning: {filepath} already exists")
        exit(1)
    
    # Combine all results
    full_results = {
        'config': {
            'dataset': args.dataset,
            'filter_method': filter_method_name,
            'ranking_method': ranking_method_name,
            'stage': args.stage,
            'bm25_topk': args.bm25_topk,
            'filter_topk': args.filter_topk,
        },
        'metrics': metrics_dict,
        'codecarbon': codecarbon_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=4)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def save_rundict(rundict: Dict, output_path: str):
    """Persist rundict to disk for later reranking."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rundict, f, indent=2)
    print(f"Rundict saved to: {output_path}")


# ========================================
# Main Execution
# ========================================


def main():
    """Main execution function."""
    args = get_args()
    if args.custom_model in ("zephyr", "vicuna") and args.listwise_model == "custom_llm" :
        args.listwise_model = args.custom_model  # Use same model for listwise if specified as custom
    print(f"Configuration: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts
    prompts_file = 'prompts.json'
    if not os.path.exists(prompts_file):
        print(f"Error: {prompts_file} not found")
        return
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)
    
    # Get prompts for dataset
    dataset_key = args.dataset
    if dataset_key.startswith('trec'):
        dataset_key = 'trec'
    
    if dataset_key not in all_prompts:
        print(f"Warning: No prompts found for {dataset_key}, using 'scifact' as default")
        dataset_key = 'scifact'
    
    prompts = all_prompts[dataset_key]
    
    # Replace --NUMBER placeholders
    prompts['S-prompt'] = prompts['S-prompt'].replace('--NUMBER', str(args.filter_topk))
    prompts['U-prompt-3'] = prompts['U-prompt-3'].replace('--NUMBER', str(args.filter_topk))
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    try:
        _, _, qrels = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is available in datasets/ directory")
        return
    
    # Load BM25 results (needed for filtering stage)
    bm25_rundict, queries_with_passages = load_bm25_results(args.dataset, args.bm25_topk)

    if args.stage in ("filter", "both") and not bm25_rundict:
        print("\nError: BM25 results not available.")
        print("Please pre-compute BM25 results using pyserini or provide cached results.")
        print("Example: Use pyserini to create BM25 index and retrieve top-100 docs for each query")
        return

    emissions_dir = os.path.join(args.output_dir, 'emissions')
    os.makedirs(emissions_dir, exist_ok=True)

    if args.stage == "rerank":
        if args.load_filtered_rundict:
            filter_method_name = os.path.basename(args.load_filtered_rundict).split("_")[2]  # Extract filter method from filename
            filter_number = f"filter{args.filter_topk}"
        else:
            filter_method_name = "Direct"
            filter_number = f"filter{args.bm25_topk}"
    elif args.filter_method == "custom_llm":
        filter_method_name = f"CustomLLM-{args.custom_model}"
        filter_number = f"filter{args.filter_topk}"
    elif args.filter_method == "custom_llm_vllm":
        filter_method_name = f"CustomLLM-vLLM-{args.custom_model}"
        filter_number = f"filter{args.filter_topk}"
    elif args.filter_method == "pointwise":
        filter_method_name = "Pointwise"
        filter_number = f"filter{args.bm25_topk}"
    elif args.filter_method == "bert_index":
        filter_method_name = f"BERT-{args.bert_model}"
        filter_number = f"filter{args.bm25_topk}"
    elif args.filter_method == "RankLLM_custom":
        filter_method_name = f"RankLLM-{args.custom_model}"
        filter_number = f"filter{args.filter_topk}"

    if args.stage == "filter":
        ranking_method_name = "No-Rerank"
    elif args.ranking_method == "custom_llm":
        ranking_method_name = f"CustomLLM-{args.custom_model}"
    elif args.ranking_method == "custom_llm_vllm":
        ranking_method_name = f"CustomLLM-vLLM-{args.custom_model}"
    elif args.ranking_method == "listwise":
        if args.listwise_model == "custom_llm":
            ranking_method_name = f"Listwise-{args.custom_model}-W{args.listwise_window_size}-S{args.listwise_stride}"
        else:
            ranking_method_name = f"Listwise-{args.listwise_model}-W{args.listwise_window_size}-S{args.listwise_stride}"

    filter_emissions_file = (
        f"emissions_filter_{args.dataset}_{filter_method_name}_{filter_number}_top{args.bm25_topk}.csv"
    )
    ranking_emissions_file = (
        f"emissions_ranking_{args.dataset}_{ranking_method_name}_filter{args.filter_topk}_top{args.bm25_topk}.csv"
    )

    tracker_filter = None
    tracker_ranking = None

    filtered_rundict = None
    filter_engine = None

    queries_to_process = [qid for qid in qrels.keys() if qid in queries_with_passages]
    print(f"Processing {len(queries_to_process)} queries")

    filtered_output_path = os.path.join(
        args.output_dir,
        "rundicts",
        f"filtered_{args.dataset}_{filter_method_name}_{filter_number}_top{args.bm25_topk}.json"
    )

    ranking_output_path = os.path.join(
        args.output_dir,
        "rundicts",
        f"ranked_{args.dataset}_{ranking_method_name}_filter{args.filter_topk}_top{args.bm25_topk}.json"
    )

    try:
        with open(ranking_output_path, 'r', encoding='utf-8') as f:
            rundict_rerank = json.load(f)
    except FileNotFoundError:
        print(f"Error: Reranked results not found at {ranking_output_path}")
        return
    
    try:        
        with open(filtered_output_path, 'r', encoding='utf-8') as f:
            filtered_rundict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Filtered rundict not found at {filtered_output_path}")
        return

    print("\nEvaluating results...")

    bm25_metrics = evaluate_metrics(qrels, bm25_rundict) if bm25_rundict else {}
    if bm25_metrics:
        print("BM25 baseline metrics:")
        print(json.dumps(bm25_metrics, indent=2))
    filtered_metrics = evaluate_metrics(qrels, filtered_rundict) if filtered_rundict else {}
    if filtered_metrics:
        print("\nFiltered results metrics:")
        print(json.dumps(filtered_metrics, indent=2))

    if args.stage in ("both", "rerank"):
        rerank_metrics = evaluate_metrics(qrels, rundict_rerank)
        print("\nTwo-stage reranking metrics:")
        print(json.dumps(rerank_metrics, indent=2))
    else:
        rerank_metrics = {}

    codecarbon_metrics = {}

    filter_csv_path = os.path.join(emissions_dir, filter_emissions_file)
    if os.path.exists(filter_csv_path):
        with open(filter_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            reader_list = list(reader)
            filter_data = reader_list[-1] if reader_list else {}
            codecarbon_metrics['filter'] = {
                'emissions_kg': float(filter_data.get('emissions', 0)),
                'energy_consumed_kwh': float(filter_data.get('energy_consumed', 0)),
                'duration_s': float(filter_data.get('duration', 0)),
            }

    if tracker_ranking:
        ranking_csv_path = os.path.join(emissions_dir, ranking_emissions_file)
        if os.path.exists(ranking_csv_path):
            with open(ranking_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                reader_list = list(reader)
                ranking_data = reader_list[-1] if reader_list else {}
                codecarbon_metrics['ranking'] = {
                    'emissions_kg': float(ranking_data.get('emissions', 0)),
                    'energy_consumed_kwh': float(ranking_data.get('energy_consumed', 0)),
                    'duration_s': float(ranking_data.get('duration', 0)),
                }

    # timing_metrics = {}
    # if filter_avg_time is not None and filter_total_time is not None:
    #     timing_metrics['filter_avg_time_s'] = filter_avg_time
    #     timing_metrics['filter_total_time_s'] = filter_total_time
    # else:
    #     timing_metrics['filter_avg_time_s'] = 0
    #     timing_metrics['filter_total_time_s'] = 0

    # if ranking_engine:
    #     timing_metrics['ranking_avg_time_s'] = ranking_engine.ranking_time / max(ranking_engine.ranking_count, 1)
    #     timing_metrics['ranking_total_time_s'] = ranking_engine.ranking_time
    # else:
    #     timing_metrics['ranking_avg_time_s'] = 0
    #     timing_metrics['ranking_total_time_s'] = 0
    # codecarbon_metrics['timing'] = timing_metrics

    print("\nCodeCarbon metrics:")
    print(json.dumps(codecarbon_metrics, indent=2))

    all_metrics = {
        'bm25': bm25_metrics,
        'filtered': filtered_metrics,
        'two_stage_reranking': rerank_metrics,
    }

    results_file = save_results(args, all_metrics, codecarbon_metrics, args.output_dir, filter_method_name, ranking_method_name)

    print("\n" + "="*20)
    print("EVALUATION COMPLETE")
    print("="*20)
    print(f"Dataset: {args.dataset}")
    print(f"Filter method: {filter_method_name}")
    print(f"Ranking method: {ranking_method_name}")
    print(f"Results saved to: {results_file}")
    print("="*20)


if __name__ == '__main__':
    main()
