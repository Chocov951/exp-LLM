# Complete Guide: all_in_one_rankllm.py - Ranking Evaluation Pipeline

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [File Structure](#file-structure)
4. [Installation](#installation)
5. [Usage Examples](#usage-examples)
6. [Comparing Strategies](#comparing-strategies)
7. [Advanced Configuration](#advanced-configuration)
8. [Output Format](#output-format)
9. [Troubleshooting](#troubleshooting)

## Overview

This implementation provides a complete pipeline for evaluating ranking strategies with a novel two-stage approach. It follows the specifications in `RANK_LLM_INTEGRATION_GUIDE.md` and enables:

- **Multiple dataset support**: Any BEIR-format dataset
- **Flexible ranking strategies**: Compare existing methods from rank_llm with custom two-stage approaches
- **Environmental tracking**: Monitor CO2 emissions and energy consumption with CodeCarbon
- **Comprehensive metrics**: NDCG, MRR, Recall, Precision at multiple cutoffs
- **Easy comparison**: Built-in tools to compare different ranking strategies

### Two-Stage Ranking Strategy

**Stage 1: Filter**
- Reduces initial candidate set (e.g., BM25 top-100) to smaller set (default: top-10)
- Methods: Custom LLM prompts, BERT embeddings, RankT5

**Stage 2: Ranking**
- Fine-grained ranking of filtered passages
- Methods: Custom LLM prompts, RankLLM listwise, RankLLM pointwise

## Quick Start

### 1. Setup

```bash
# Make setup script executable and run it
chmod +x setup_rankllm.sh
./setup_rankllm.sh
```

This will:
- Install dependencies
- Create necessary directories
- Download sample dataset (SciFact)
- Verify installation

### 2. Prepare Data

You need BM25 results for your dataset. Either:

**Option A: Use pre-computed results**
```bash
# Place in rundicts/
rundicts/rundict_scifact_bm25.json
rundicts/queries_test_scifact_bm25.json
```

**Option B: Compute with Pyserini**
```bash
python3 -m pyserini.search.lucene \
    --index beir-v1.0.0-scifact-flat \
    --topics beir-v1.0.0-scifact-test \
    --output rundict_scifact_bm25.txt \
    --bm25 --hits 100
```

### 3. Run Your First Experiment

```bash
# Basic run with BERT filtering and custom LLM ranking
python3 all_in_one_rankllm.py \
    --dataset scifact \
    --filter_method bert_index \
    --ranking_method custom_llm \
    --model_name qwen3 \
    --filter_topk 10
```

## File Structure

```
exp-LLM/
├── all_in_one_rankllm.py          # Main pipeline script
├── all_in_one_rankllm_README.md   # Detailed documentation
├── GUIDE_rankllm.md               # This file
├── example_rankllm_usage.py       # Example usage with multiple configs
├── compare_ranking_strategies.py  # Results comparison tool
├── setup_rankllm.sh              # Setup script
├── requirements_rankllm.txt       # Dependencies
├── test_rankllm.py               # Basic tests
├── prompts.json                  # Custom prompts for LLM
├── datasets/                     # BEIR format datasets
│   └── scifact/
├── rundicts/                     # BM25 and intermediate results
│   ├── rundict_scifact_bm25.json
│   └── queries_test_scifact_bm25.json
├── rankllm_results/             # Output directory
│   ├── emissions/               # CodeCarbon tracking
│   └── *.json                   # Result files
└── models/                      # Model files (if using local models)
    └── Qwen2.5-3B-Instruct/
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM (32GB+ recommended for larger models)

### Step-by-Step Installation

```bash
# 1. Install core dependencies
pip install torch transformers tqdm

# 2. Install ranking libraries
pip install pyserini ranx beir

# 3. Install environmental tracking
pip install codecarbon

# 4. Install model optimization (optional but recommended)
pip install bitsandbytes accelerate

# 5. Install rank_llm (optional, for rank_llm methods)
git clone https://github.com/castorini/rank_llm.git
cd rank_llm
pip install -e .
cd ..
```

Or use the provided requirements file:

```bash
pip install -r requirements_rankllm.txt
```

## Usage Examples

### Example 1: Compare Filter Methods

```bash
# Test different filter methods with same ranking
python3 all_in_one_rankllm.py --dataset scifact --filter_method custom_llm --ranking_method custom_llm
python3 all_in_one_rankllm.py --dataset scifact --filter_method bert_index --ranking_method custom_llm
python3 all_in_one_rankllm.py --dataset scifact --filter_method rankT5 --ranking_method custom_llm
```

### Example 2: Compare Ranking Methods

```bash
# Test different ranking methods with same filter
python3 all_in_one_rankllm.py --dataset scifact --filter_method bert_index --ranking_method custom_llm
python3 all_in_one_rankllm.py --dataset scifact --filter_method bert_index --ranking_method rankllm_listwise
python3 all_in_one_rankllm.py --dataset scifact --filter_method bert_index --ranking_method rankllm_pointwise
```

### Example 3: Test Different Models

```bash
# Compare different LLM sizes
python3 all_in_one_rankllm.py --dataset scifact --model_name qwen3
python3 all_in_one_rankllm.py --dataset scifact --model_name qwen14
python3 all_in_one_rankllm.py --dataset scifact --model_name qwen32
```

### Example 4: Vary Filter Top-K

```bash
# Test different filter thresholds
python3 all_in_one_rankllm.py --dataset scifact --filter_topk 5
python3 all_in_one_rankllm.py --dataset scifact --filter_topk 10
python3 all_in_one_rankllm.py --dataset scifact --filter_topk 20
```

### Example 5: Multiple Datasets

```bash
# Run on different datasets
python3 all_in_one_rankllm.py --dataset scifact
python3 all_in_one_rankllm.py --dataset trec-covid
python3 all_in_one_rankllm.py --dataset fiqa
python3 all_in_one_rankllm.py --dataset nfcorpus
```

### Example 6: Using the Example Script

```bash
# Run pre-configured experiments
python3 example_rankllm_usage.py

# Follow prompts to:
# - Select specific experiments
# - Run all experiments
# - Compare existing results
```

## Comparing Strategies

After running experiments, use the comparison tool:

```bash
# Basic comparison
python3 compare_ranking_strategies.py

# Specify different metric
python3 compare_ranking_strategies.py --metric recall@10

# Export detailed summary
python3 compare_ranking_strategies.py --export my_comparison.json
```

This generates:
- **Comparison tables** for each metric
- **Environmental impact** analysis
- **Efficiency analysis** (quality vs. cost)
- **Best configurations** for different criteria
- **JSON export** for further analysis

### Sample Output

```
================================================================================================
Comparison Table: NDCG@10
================================================================================================
Configuration                            BM25 Baseline   Two-Stage       Improvement     % Gain
----------------------------------------------------------------------------------------------------
bert_index → custom_llm                  0.6789          0.7456          +0.0667         +9.82%
custom_llm → custom_llm                  0.6789          0.7234          +0.0445         +6.55%
rankT5 → rankllm_listwise               0.6789          0.7123          +0.0334         +4.92%
----------------------------------------------------------------------------------------------------
```

## Advanced Configuration

### Custom Prompts

Edit `prompts.json` to customize LLM behavior:

```json
{
  "scifact": {
    "S-prompt": "Your custom system prompt...",
    "U-prompt-1": "User prompt part 1...",
    "U-prompt-2": "User prompt part 2...",
    "U-prompt-3": "User prompt part 3...",
    "rerank-S-prompt": "Reranking system prompt...",
    ...
  }
}
```

The `--NUMBER` placeholder is automatically replaced with `--filter_topk` value.

### Using Different BERT Models

```bash
python3 all_in_one_rankllm.py \
    --filter_method bert_index \
    --bert_model sentence-transformers/all-MiniLM-L6-v2
```

### Using RankLLM Models

```bash
python3 all_in_one_rankllm.py \
    --ranking_method rankllm_listwise \
    --rankllm_model castorini/rank_zephyr_7b_v1_full
```

### Batch Processing Multiple Configurations

Create a shell script:

```bash
#!/bin/bash
# run_experiments.sh

datasets=("scifact" "trec-covid" "fiqa")
filter_methods=("bert_index" "custom_llm")
ranking_methods=("custom_llm" "rankllm_listwise")

for dataset in "${datasets[@]}"; do
    for filter in "${filter_methods[@]}"; do
        for ranking in "${ranking_methods[@]}"; do
            echo "Running: $dataset - $filter → $ranking"
            python3 all_in_one_rankllm.py \
                --dataset "$dataset" \
                --filter_method "$filter" \
                --ranking_method "$ranking"
        done
    done
done

# Compare all results
python3 compare_ranking_strategies.py --export comparison_all.json
```

## Output Format

### Result File Naming

Results are automatically saved with descriptive names:

```
{dataset}_{model}_{filter-method}_{rank-method}_topk{bm25_topk}_filtertopk{filter_topk}_results.json
```

Example:
```
scifact_qwen3_filter-bert_index_rank-custom_llm_topk100_filtertopk10_results.json
```

### JSON Structure

```json
{
  "config": {
    "dataset": "scifact",
    "model_name": "qwen3",
    "filter_method": "bert_index",
    "ranking_method": "custom_llm",
    "bm25_topk": 100,
    "filter_topk": 10
  },
  "metrics": {
    "bm25": {
      "ndcg@1": 0.5234,
      "ndcg@5": 0.6123,
      "ndcg@10": 0.6789,
      "recall@10": 0.8234,
      "mrr@10": 0.7123
    },
    "two_stage_reranking": {
      "ndcg@1": 0.6123,
      "ndcg@5": 0.6891,
      "ndcg@10": 0.7456,
      "recall@10": 0.8567,
      "mrr@10": 0.7891
    }
  },
  "codecarbon": {
    "filter": {
      "emissions_kg": 0.001234,
      "energy_consumed_kwh": 0.004567,
      "duration_s": 123.45
    },
    "ranking": {
      "emissions_kg": 0.000789,
      "energy_consumed_kwh": 0.002345,
      "duration_s": 67.89
    },
    "timing": {
      "filter_avg_time_s": 1.23,
      "ranking_avg_time_s": 0.68,
      "filter_total_time_s": 123.45,
      "ranking_total_time_s": 67.89
    }
  },
  "timestamp": "2026-01-23 12:38:57"
}
```

## Troubleshooting

### Issue: "rank_llm not installed"

**Solution**: The script works without rank_llm by using fallback implementations. To use rank_llm methods:

```bash
git clone https://github.com/castorini/rank_llm.git
cd rank_llm
pip install -e .
```

### Issue: "BM25 results not available"

**Solution**: Pre-compute BM25 results using Pyserini:

```bash
# Install Pyserini
pip install pyserini

# Download and index dataset
python3 -c "
from pyserini.search import LuceneSearcher
searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-scifact-flat')
"

# Run retrieval (see documentation for details)
```

### Issue: "Out of memory"

**Solutions**:
1. Use smaller model: `--model_name qwen3`
2. Reduce batch sizes: `--bm25_topk 50 --filter_topk 5`
3. Ensure 8-bit quantization is enabled (default)
4. Close other applications

### Issue: "Missing prompts for dataset"

**Solution**: The script falls back to 'scifact' prompts. To add custom prompts:

1. Edit `prompts.json`
2. Add section for your dataset
3. Include all required prompt keys

### Issue: "Slow performance"

**Solutions**:
1. Use BERT filtering (faster than LLM): `--filter_method bert_index`
2. Pre-compute and cache results
3. Use GPU acceleration
4. Consider using vLLM backend (future enhancement)

### Issue: "Different results than expected"

**Possible causes**:
1. Different prompts than original implementation
2. Different model versions
3. Different random seeds (if using sampling)

**Check**:
- Verify prompts match expected format
- Compare with BM25 baseline
- Check model paths and versions

## Performance Tips

### Speed Optimization

1. **Use caching**: Results are cached in `rundicts/`
2. **BERT for filtering**: Faster than LLM filtering
3. **Smaller models**: qwen3 is much faster than qwen72
4. **Pre-compute BM25**: Don't recompute for each run

### Memory Optimization

1. **8-bit quantization**: Enabled by default
2. **Reduce topk values**: Fewer passages to process
3. **Batch processing**: Process queries in batches
4. **Clear cache**: `torch.cuda.empty_cache()` between runs

### Quality Optimization

1. **Larger models**: qwen14/32 often better than qwen3
2. **Two-stage approach**: Usually beats single-stage
3. **Custom prompts**: Tune for your domain
4. **Higher filter_topk**: More passages for ranking stage

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{your-paper,
    title={Two-Stage LLM Reranking with Environmental Tracking},
    author={Your Name},
    booktitle={Conference},
    year={2026}
}
```

## Related Resources

- **RANK_LLM_INTEGRATION_GUIDE.md**: Detailed integration guide
- **all_in_one_rankllm_README.md**: Technical documentation
- [RankLLM GitHub](https://github.com/castorini/rank_llm)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [Pyserini](https://github.com/castorini/pyserini)
- [CodeCarbon](https://github.com/mlco2/codecarbon)

## Support

For issues or questions:
1. Check this guide and README
2. Review existing issues in repository
3. Create new issue with detailed description

---

**Happy Ranking! 🚀**
