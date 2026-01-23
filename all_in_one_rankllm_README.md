# all_in_one_rankllm.py - Complete Ranking Evaluation Pipeline

## Overview

This script implements a comprehensive ranking evaluation pipeline using the rank_llm library with a novel two-stage ranking strategy. It provides:

1. **Full ranking evaluation pipeline** using rank_llm library
2. **CodeCarbon tracking** for environmental cost monitoring
3. **Support for multiple datasets** from rank_llm/BEIR
4. **Top-100 docs from pyserini BM25** for each dataset
5. **Parametrable two-stage ranking strategy**:
   - Filter step: custom LLM & prompt, rankT5, or BERT index
   - Ranking step: custom LLM & prompt or rank_llm existing methods
6. **Comprehensive metrics export** (NDCG, MRR, Recall, CodeCarbon) to JSON

## Features

### Two-Stage Ranking Strategy

#### Stage 1: Filter
Reduces the initial candidate set (e.g., top-100 from BM25) to a smaller set of most relevant passages (default: top-10).

**Filter Methods:**
- `custom_llm`: Use custom LLM with prompts from prompts.json
- `bert_index`: Use BERT embeddings for semantic similarity
- `rankT5`: Use RankT5 model from rank_llm (if available)

#### Stage 2: Ranking
Performs fine-grained ranking of the filtered passages.

**Ranking Methods:**
- `custom_llm`: Use custom LLM with prompts from prompts.json
- `rankllm_listwise`: Use rank_llm's listwise ranking approach
- `rankllm_pointwise`: Use rank_llm's pointwise ranking approach

### Metrics Tracked

**Ranking Metrics:**
- NDCG@1, @3, @5, @10, @20
- Recall@1, @5, @10, @20, @100
- MRR@10, @100
- Precision@1, @5, @10

**Environmental Metrics (CodeCarbon):**
- CO2 emissions (kg)
- Energy consumed (kWh)
- Duration (seconds)
- Per-stage and per-query metrics

## Installation

### Required Dependencies

```bash
# Install core dependencies
pip install torch transformers
pip install pyserini
pip install ranx
pip install beir
pip install codecarbon
pip install tqdm

# Install rank_llm (if available)
pip install rank-llm
# OR install from source
git clone https://github.com/castorini/rank_llm.git
cd rank_llm
pip install -e .
```

### Optional Dependencies

```bash
# For BERT filtering
pip install sentence-transformers

# For GPU acceleration
pip install bitsandbytes  # For 8-bit quantization
```

## Usage

### Basic Usage

```bash
# Run with default configuration (SciFact dataset, custom LLM for both stages)
python all_in_one_rankllm.py --dataset scifact

# Specify filter and ranking methods
python all_in_one_rankllm.py \
    --dataset scifact \
    --filter_method bert_index \
    --ranking_method custom_llm \
    --filter_topk 10
```

### Advanced Configuration

```bash
# Use BERT for filtering, rank_llm for ranking
python all_in_one_rankllm.py \
    --dataset trec-covid \
    --filter_method bert_index \
    --ranking_method rankllm_listwise \
    --bm25_topk 100 \
    --filter_topk 20 \
    --model_name qwen14 \
    --output_dir my_results

# Use different models for different stages
python all_in_one_rankllm.py \
    --dataset fiqa \
    --filter_method custom_llm \
    --ranking_method custom_llm \
    --model_name qwen3 \
    --bert_model BAAI/bge-m3 \
    --filter_topk 15
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | scifact | Dataset to evaluate (scifact, trec-covid, fiqa, etc.) |
| `--model_name` | str | qwen3 | Model for custom LLM (qwen3/14/32/72, calme, llama, mistral) |
| `--bm25_topk` | int | 100 | Number of top documents from BM25 to rerank |
| `--filter_method` | str | custom_llm | Method for filter stage (custom_llm, rankT5, bert_index) |
| `--filter_topk` | int | 10 | Number of passages to keep after filtering |
| `--ranking_method` | str | custom_llm | Method for ranking stage (custom_llm, rankllm_listwise, rankllm_pointwise) |
| `--rankllm_model` | str | castorini/rank_zephyr_7b_v1_full | RankLLM model if using rank_llm methods |
| `--bert_model` | str | BAAI/bge-m3 | BERT model for BERT index filtering |
| `--output_dir` | str | rankllm_results | Directory to save results |
| `--use_cache` | flag | False | Use cached results if available |

## Output Format

### Results JSON Structure

Results are saved with descriptive filenames containing all hyperparameters:

**Filename format:**
```
{dataset}_{model}_{filter-method}_{rank-method}_topk{bm25_topk}_filtertopk{filter_topk}_results.json
```

**Example:**
```
scifact_qwen3_filter-bert_index_rank-custom_llm_topk100_filtertopk10_results.json
```

**JSON structure:**
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
            "ndcg@10": 0.6789,
            "recall@10": 0.8234,
            "mrr@10": 0.7123
        },
        "two_stage_reranking": {
            "ndcg@10": 0.7456,
            "recall@10": 0.8567,
            "mrr@10": 0.7891
        }
    },
    "codecarbon": {
        "filter": {
            "emissions_kg": 0.0012,
            "energy_consumed_kwh": 0.0045,
            "duration_s": 123.45
        },
        "ranking": {
            "emissions_kg": 0.0008,
            "energy_consumed_kwh": 0.0023,
            "duration_s": 67.89
        },
        "timing": {
            "filter_avg_time_s": 1.23,
            "ranking_avg_time_s": 0.68
        }
    },
    "timestamp": "2026-01-23 12:38:57"
}
```

## Data Requirements

### Dataset Structure

The script expects datasets in BEIR format:
```
datasets/
├── scifact/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       └── test.tsv
├── trec-covid/
└── fiqa/
```

### BM25 Results

Pre-computed BM25 results should be in:
```
rundicts/
├── rundict_{dataset}_bm25.json
└── queries_test_{dataset}_bm25.json
```

**Format:**
```json
{
    "query_id": {
        "doc_id_1": score_1,
        "doc_id_2": score_2,
        ...
    }
}
```

### Prompts Configuration

The script uses `prompts.json` for custom LLM prompts. Each dataset should have:
- `S-prompt`: System prompt for filtering
- `U-prompt-1`, `U-prompt-2`, `U-prompt-3`: User prompts for filtering
- `rerank-S-prompt`: System prompt for ranking
- `rerank-U-prompt-1`, `rerank-U-prompt-2`, `rerank-U-prompt-3`: User prompts for ranking

The `--NUMBER` placeholder in prompts is automatically replaced with `filter_topk`.

## Comparison with Existing Methods

This implementation allows easy comparison between:

1. **Baseline methods:**
   - BM25 (sparse retrieval)
   - BERT semantic search

2. **Single-stage methods:**
   - RankLLM listwise
   - RankLLM pointwise
   - Custom LLM ranking

3. **Two-stage methods (novel):**
   - BERT → Custom LLM
   - Custom LLM → Custom LLM
   - BERT → RankLLM
   - RankT5 → RankLLM
   - Any combination of filter and ranking methods

## Performance Optimization

### Memory Management

For large models, use quantization:
```python
# The script automatically uses 8-bit quantization for supported models
# Configured in create_model_config()
```

### Batch Processing

The script processes queries sequentially. For faster processing:
- Pre-compute BM25 results offline
- Use GPU with sufficient memory
- Consider using rank_llm's vLLM backend (future enhancement)

### Caching

Results are cached in:
- `rundicts/` - BM25 and intermediate results
- `{output_dir}/emissions/` - CodeCarbon tracking files

## Troubleshooting

### Common Issues

**1. rank_llm not installed:**
- The script will fall back to custom LLM implementations
- Install rank_llm from source or PyPI for full functionality

**2. BM25 results not available:**
- Pre-compute using pyserini:
  ```bash
  python -m pyserini.search.lucene \
      --index {index_path} \
      --topics {queries_file} \
      --output {output_file} \
      --bm25
  ```

**3. Out of memory:**
- Reduce `bm25_topk` and `filter_topk`
- Use smaller model (qwen3 instead of qwen72)
- Enable 8-bit quantization (default)

**4. Missing datasets:**
- Download BEIR datasets:
  ```python
  from beir import util
  util.download_and_unzip(
      "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
      "datasets"
  )
  ```

## Integration with RankLLM

This implementation follows the [RANK_LLM_INTEGRATION_GUIDE.md](RANK_LLM_INTEGRATION_GUIDE.md) and provides:

- **Hybrid approach**: Uses rank_llm infrastructure where available, falls back to custom implementations
- **Custom prompts**: Maintains ability to use domain-specific prompts from prompts.json
- **Two-stage logic**: Implements novel research contribution as first-class feature
- **Comparison framework**: Easy A/B testing between methods

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{your-paper,
    title={Two-Stage LLM Reranking for Information Retrieval},
    author={Your Name},
    booktitle={ECIR},
    year={2026}
}
```

## Related Work

- [RankLLM](https://github.com/castorini/rank_llm) - Castorini's ranking library
- [BEIR](https://github.com/beir-cellar/beir) - Benchmark for IR
- [Pyserini](https://github.com/castorini/pyserini) - Python toolkit for reproducible IR
- [CodeCarbon](https://github.com/mlco2/codecarbon) - Track CO2 emissions

## License

This code follows the same license as the parent repository.
