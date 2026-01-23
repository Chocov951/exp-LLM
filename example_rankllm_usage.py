#!/usr/bin/env python3
"""
Example usage script for all_in_one_rankllm.py

This script demonstrates how to use the ranking pipeline with different configurations.
"""

import subprocess
import os

# Example configurations to test
CONFIGS = [
    {
        "name": "SciFact - Custom LLM (Both Stages)",
        "args": [
            "--dataset", "scifact",
            "--filter_method", "custom_llm",
            "--ranking_method", "custom_llm",
            "--model_name", "qwen3",
            "--bm25_topk", "100",
            "--filter_topk", "10"
        ]
    },
    {
        "name": "SciFact - BERT Filter + Custom LLM Ranking",
        "args": [
            "--dataset", "scifact",
            "--filter_method", "bert_index",
            "--ranking_method", "custom_llm",
            "--model_name", "qwen3",
            "--bert_model", "BAAI/bge-m3",
            "--bm25_topk", "100",
            "--filter_topk", "10"
        ]
    },
    {
        "name": "TREC-COVID - Custom LLM with 20 Filter Top-K",
        "args": [
            "--dataset", "trec-covid",
            "--filter_method", "custom_llm",
            "--ranking_method", "custom_llm",
            "--model_name", "qwen3",
            "--bm25_topk", "100",
            "--filter_topk", "20"
        ]
    },
]


def run_experiment(config):
    """Run a single experiment configuration."""
    print("\n" + "="*80)
    print(f"Running: {config['name']}")
    print("="*80)
    
    cmd = ["python3", "all_in_one_rankllm.py"] + config["args"]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✓ {config['name']} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {config['name']} failed")
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def compare_results(output_dir="rankllm_results"):
    """Compare results from different configurations."""
    import json
    import glob
    
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    result_files = glob.glob(os.path.join(output_dir, "*_results.json"))
    
    if not result_files:
        print("No result files found")
        return
    
    results = []
    for filepath in result_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append({
                'file': os.path.basename(filepath),
                'config': data['config'],
                'metrics': data['metrics'],
                'codecarbon': data['codecarbon']
            })
    
    # Print comparison table
    print("\nNDCG@10 Comparison:")
    print("-" * 80)
    print(f"{'Configuration':<50} {'BM25':<10} {'Two-Stage':<10} {'Improvement':<10}")
    print("-" * 80)
    
    for result in results:
        config_name = f"{result['config']['filter_method']} -> {result['config']['ranking_method']}"
        bm25_ndcg = result['metrics']['bm25'].get('ndcg@10', 0)
        rerank_ndcg = result['metrics']['two_stage_reranking'].get('ndcg@10', 0)
        improvement = ((rerank_ndcg - bm25_ndcg) / bm25_ndcg * 100) if bm25_ndcg > 0 else 0
        
        print(f"{config_name:<50} {bm25_ndcg:<10.4f} {rerank_ndcg:<10.4f} {improvement:<10.2f}%")
    
    print("-" * 80)
    
    # Print environmental impact
    print("\nEnvironmental Impact (CO2 emissions in kg):")
    print("-" * 80)
    print(f"{'Configuration':<50} {'Filter':<10} {'Ranking':<10} {'Total':<10}")
    print("-" * 80)
    
    for result in results:
        config_name = f"{result['config']['filter_method']} -> {result['config']['ranking_method']}"
        filter_co2 = result['codecarbon'].get('filter', {}).get('emissions_kg', 0)
        ranking_co2 = result['codecarbon'].get('ranking', {}).get('emissions_kg', 0)
        total_co2 = filter_co2 + ranking_co2
        
        print(f"{config_name:<50} {filter_co2:<10.6f} {ranking_co2:<10.6f} {total_co2:<10.6f}")
    
    print("-" * 80)


def main():
    """Main function to run examples."""
    print("="*80)
    print("ALL_IN_ONE_RANKLLM.PY - EXAMPLE USAGE")
    print("="*80)
    print()
    print("This script demonstrates different configurations of the ranking pipeline.")
    print("You can modify CONFIGS in this script to test your own configurations.")
    print()
    
    # Check if script exists
    if not os.path.exists("all_in_one_rankllm.py"):
        print("Error: all_in_one_rankllm.py not found in current directory")
        return
    
    # Ask user which experiments to run
    print("Available experiments:")
    for i, config in enumerate(CONFIGS, 1):
        print(f"{i}. {config['name']}")
    print(f"{len(CONFIGS)+1}. Run all experiments")
    print(f"{len(CONFIGS)+2}. Just compare existing results")
    print()
    
    try:
        choice = input("Enter your choice (1-{}): ".format(len(CONFIGS)+2))
        choice = int(choice)
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")
        return
    
    if choice == len(CONFIGS) + 2:
        # Just compare results
        compare_results()
        return
    elif choice == len(CONFIGS) + 1:
        # Run all
        configs_to_run = CONFIGS
    elif 1 <= choice <= len(CONFIGS):
        # Run specific one
        configs_to_run = [CONFIGS[choice - 1]]
    else:
        print("Invalid choice")
        return
    
    # Run experiments
    success_count = 0
    for config in configs_to_run:
        if run_experiment(config):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"Completed {success_count}/{len(configs_to_run)} experiments successfully")
    print("="*80)
    
    # Compare results if multiple experiments were run
    if success_count > 0:
        compare_results()


if __name__ == '__main__':
    main()
