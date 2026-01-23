#!/usr/bin/env python3
"""
Compare ranking strategies using all_in_one_rankllm.py results.

This script loads results from multiple experiments and creates comparison tables,
charts, and analysis to help identify the best ranking strategy.
"""

import os
import json
import glob
from typing import List, Dict
import argparse


def load_results(results_dir: str) -> List[Dict]:
    """Load all result JSON files from a directory."""
    result_files = glob.glob(os.path.join(results_dir, "*_results.json"))
    
    results = []
    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['filepath'] = filepath
                data['filename'] = os.path.basename(filepath)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def create_comparison_table(results: List[Dict], metric: str = 'ndcg@10'):
    """Create comparison table for a specific metric."""
    print(f"\n{'='*100}")
    print(f"Comparison Table: {metric.upper()}")
    print(f"{'='*100}")
    
    # Table header
    header_format = "{:<40} {:<15} {:<15} {:<15} {:<15}"
    print(header_format.format(
        "Configuration",
        "BM25 Baseline",
        "Two-Stage",
        "Improvement",
        "% Gain"
    ))
    print("-" * 100)
    
    # Sort by two-stage performance
    results_sorted = sorted(
        results,
        key=lambda x: x.get('metrics', {}).get('two_stage_reranking', {}).get(metric, 0),
        reverse=True
    )
    
    for result in results_sorted:
        config = result['config']
        config_name = f"{config['filter_method']} → {config['ranking_method']}"
        
        bm25_score = result['metrics'].get('bm25', {}).get(metric, 0)
        rerank_score = result['metrics'].get('two_stage_reranking', {}).get(metric, 0)
        improvement = rerank_score - bm25_score
        percent_gain = (improvement / bm25_score * 100) if bm25_score > 0 else 0
        
        print(header_format.format(
            config_name[:39],
            f"{bm25_score:.4f}",
            f"{rerank_score:.4f}",
            f"{improvement:+.4f}",
            f"{percent_gain:+.2f}%"
        ))
    
    print("-" * 100)


def create_metrics_summary(results: List[Dict]):
    """Create summary of all metrics."""
    print(f"\n{'='*100}")
    print("Metrics Summary")
    print(f"{'='*100}")
    
    metrics_to_show = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'recall@5', 'recall@10', 'mrr@10']
    
    for result in results:
        config = result['config']
        print(f"\nConfiguration: {config['filter_method']} → {config['ranking_method']}")
        print(f"Dataset: {config['dataset']}, Model: {config['model_name']}")
        print("-" * 100)
        
        print(f"{'Metric':<15} {'BM25':<12} {'Two-Stage':<12} {'Improvement':<12}")
        print("-" * 50)
        
        for metric in metrics_to_show:
            bm25_val = result['metrics'].get('bm25', {}).get(metric, 0)
            rerank_val = result['metrics'].get('two_stage_reranking', {}).get(metric, 0)
            improvement = rerank_val - bm25_val
            
            print(f"{metric:<15} {bm25_val:<12.4f} {rerank_val:<12.4f} {improvement:+12.4f}")


def create_environmental_summary(results: List[Dict]):
    """Create summary of environmental impact."""
    print(f"\n{'='*100}")
    print("Environmental Impact Summary (CodeCarbon)")
    print(f"{'='*100}")
    
    header_format = "{:<40} {:<15} {:<15} {:<15} {:<15}"
    print(header_format.format(
        "Configuration",
        "CO2 (kg)",
        "Energy (kWh)",
        "Filter Time (s)",
        "Rank Time (s)"
    ))
    print("-" * 100)
    
    for result in results:
        config = result['config']
        config_name = f"{config['filter_method']} → {config['ranking_method']}"
        
        codecarbon = result.get('codecarbon', {})
        
        filter_co2 = codecarbon.get('filter', {}).get('emissions_kg', 0)
        ranking_co2 = codecarbon.get('ranking', {}).get('emissions_kg', 0)
        total_co2 = filter_co2 + ranking_co2
        
        filter_energy = codecarbon.get('filter', {}).get('energy_consumed_kwh', 0)
        ranking_energy = codecarbon.get('ranking', {}).get('energy_consumed_kwh', 0)
        total_energy = filter_energy + ranking_energy
        
        timing = codecarbon.get('timing', {})
        filter_time = timing.get('filter_total_time_s', 0)
        ranking_time = timing.get('ranking_total_time_s', 0)
        
        print(header_format.format(
            config_name[:39],
            f"{total_co2:.6f}",
            f"{total_energy:.6f}",
            f"{filter_time:.2f}",
            f"{ranking_time:.2f}"
        ))
    
    print("-" * 100)


def create_efficiency_analysis(results: List[Dict]):
    """Analyze efficiency (quality vs. cost)."""
    print(f"\n{'='*100}")
    print("Efficiency Analysis (Quality vs Environmental Cost)")
    print(f"{'='*100}")
    
    header_format = "{:<40} {:<12} {:<15} {:<20}"
    print(header_format.format(
        "Configuration",
        "NDCG@10",
        "CO2 (kg)",
        "NDCG per kg CO2"
    ))
    print("-" * 100)
    
    for result in results:
        config = result['config']
        config_name = f"{config['filter_method']} → {config['ranking_method']}"
        
        ndcg10 = result['metrics'].get('two_stage_reranking', {}).get('ndcg@10', 0)
        
        codecarbon = result.get('codecarbon', {})
        filter_co2 = codecarbon.get('filter', {}).get('emissions_kg', 0)
        ranking_co2 = codecarbon.get('ranking', {}).get('emissions_kg', 0)
        total_co2 = filter_co2 + ranking_co2
        
        efficiency = ndcg10 / total_co2 if total_co2 > 0 else 0
        
        print(header_format.format(
            config_name[:39],
            f"{ndcg10:.4f}",
            f"{total_co2:.6f}",
            f"{efficiency:.2f}"
        ))
    
    print("-" * 100)


def find_best_configurations(results: List[Dict]):
    """Find best configurations for different criteria."""
    print(f"\n{'='*100}")
    print("Best Configurations by Different Criteria")
    print(f"{'='*100}")
    
    if not results:
        print("No results to analyze")
        return
    
    # Best NDCG@10
    best_ndcg = max(results, 
                    key=lambda x: x['metrics'].get('two_stage_reranking', {}).get('ndcg@10', 0))
    print(f"\n🏆 Best NDCG@10:")
    print(f"   Configuration: {best_ndcg['config']['filter_method']} → {best_ndcg['config']['ranking_method']}")
    print(f"   Score: {best_ndcg['metrics']['two_stage_reranking'].get('ndcg@10', 0):.4f}")
    
    # Best Recall@10
    best_recall = max(results,
                      key=lambda x: x['metrics'].get('two_stage_reranking', {}).get('recall@10', 0))
    print(f"\n🏆 Best Recall@10:")
    print(f"   Configuration: {best_recall['config']['filter_method']} → {best_recall['config']['ranking_method']}")
    print(f"   Score: {best_recall['metrics']['two_stage_reranking'].get('recall@10', 0):.4f}")
    
    # Lowest CO2
    best_co2 = min(results,
                   key=lambda x: (x['codecarbon'].get('filter', {}).get('emissions_kg', float('inf')) +
                                 x['codecarbon'].get('ranking', {}).get('emissions_kg', float('inf'))))
    total_co2 = (best_co2['codecarbon'].get('filter', {}).get('emissions_kg', 0) +
                best_co2['codecarbon'].get('ranking', {}).get('emissions_kg', 0))
    print(f"\n🌱 Lowest CO2 Emissions:")
    print(f"   Configuration: {best_co2['config']['filter_method']} → {best_co2['config']['ranking_method']}")
    print(f"   CO2: {total_co2:.6f} kg")
    
    # Best efficiency (NDCG per CO2)
    results_with_efficiency = []
    for r in results:
        ndcg10 = r['metrics'].get('two_stage_reranking', {}).get('ndcg@10', 0)
        co2 = (r['codecarbon'].get('filter', {}).get('emissions_kg', 0) +
               r['codecarbon'].get('ranking', {}).get('emissions_kg', 0))
        if co2 > 0:
            results_with_efficiency.append((r, ndcg10 / co2))
    
    if results_with_efficiency:
        best_efficiency, eff_score = max(results_with_efficiency, key=lambda x: x[1])
        print(f"\n⚡ Best Efficiency (NDCG@10 per kg CO2):")
        print(f"   Configuration: {best_efficiency['config']['filter_method']} → {best_efficiency['config']['ranking_method']}")
        print(f"   Efficiency: {eff_score:.2f}")
    
    print()


def export_summary_json(results: List[Dict], output_file: str):
    """Export summary to JSON file."""
    summary = {
        'total_experiments': len(results),
        'configurations': [],
        'best_by_metric': {},
    }
    
    for result in results:
        config_summary = {
            'filter_method': result['config']['filter_method'],
            'ranking_method': result['config']['ranking_method'],
            'dataset': result['config']['dataset'],
            'model': result['config']['model_name'],
            'metrics': result['metrics']['two_stage_reranking'],
            'environmental': {
                'total_co2_kg': (
                    result['codecarbon'].get('filter', {}).get('emissions_kg', 0) +
                    result['codecarbon'].get('ranking', {}).get('emissions_kg', 0)
                ),
                'total_energy_kwh': (
                    result['codecarbon'].get('filter', {}).get('energy_consumed_kwh', 0) +
                    result['codecarbon'].get('ranking', {}).get('energy_consumed_kwh', 0)
                ),
            }
        }
        summary['configurations'].append(config_summary)
    
    # Find best by each metric
    for metric in ['ndcg@10', 'recall@10', 'mrr@10']:
        if results:
            best = max(results, 
                      key=lambda x: x['metrics'].get('two_stage_reranking', {}).get(metric, 0))
            summary['best_by_metric'][metric] = {
                'configuration': f"{best['config']['filter_method']} → {best['config']['ranking_method']}",
                'score': best['metrics']['two_stage_reranking'].get(metric, 0)
            }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare RankLLM experiment results")
    parser.add_argument("--results_dir", type=str, default="rankllm_results",
                       help="Directory containing result JSON files")
    parser.add_argument("--metric", type=str, default="ndcg@10",
                       help="Primary metric for comparison")
    parser.add_argument("--export", type=str, default="comparison_summary.json",
                       help="Export summary to JSON file")
    
    args = parser.parse_args()
    
    print("="*100)
    print("RankLLM Experiment Comparison Tool")
    print("="*100)
    
    # Load results
    print(f"\nLoading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found. Please run experiments first.")
        return
    
    print(f"Found {len(results)} experiment results")
    
    # Create comparisons
    create_comparison_table(results, args.metric)
    create_metrics_summary(results)
    create_environmental_summary(results)
    create_efficiency_analysis(results)
    find_best_configurations(results)
    
    # Export summary
    if args.export:
        export_summary_json(results, args.export)
    
    print("\n" + "="*100)
    print("Comparison Complete!")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
