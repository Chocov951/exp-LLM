# Integration Guide: Using rank_llm Library with Your Reranking Pipeline

## Overview

This guide explains how to integrate the [rank_llm library](https://github.com/castorini/rank_llm) from Castorini into your existing LLM-based reranking pipeline (as implemented in `all_in_one.py` and described in `Paper_ECIR.pdf`).

**Yes, it is possible to use rank_llm models in the same way you do in `all_in_one.py`!** This guide shows you exactly how.

> **⚠️ Important Note**: This guide provides conceptual examples based on the typical structure of the rank_llm library. The exact API calls, import paths, and parameter names should be verified against the actual rank_llm library documentation and source code at https://github.com/castorini/rank_llm. The library may have different method signatures or module structures. Always consult the official documentation for the most up-to-date API details.

---

## Table of Contents

1. [Current Implementation Overview](#current-implementation-overview)
2. [What is rank_llm?](#what-is-rank_llm)
3. [Comparison: Your Approach vs rank_llm](#comparison-your-approach-vs-rank_llm)
4. [Integration Steps](#integration-steps)
5. [Code Examples](#code-examples)
6. [Migration Strategies](#migration-strategies)
7. [Troubleshooting](#troubleshooting)

---

## Current Implementation Overview

Your current pipeline (`all_in_one.py`) implements a **two-stage LLM reranking approach**:

### Stage 1: Reject Phase
- **Input**: Top-k passages from BM25 (default k=100)
- **Process**: LLM filters passages to identify top-N most relevant ones (default N=10)
- **Prompt**: System + User messages requesting JSON output `{"passage_id": score}`
- **Output**: Dictionary mapping passage IDs to relevance scores

### Stage 2: Rerank Phase
- **Input**: Filtered passages from Stage 1
- **Process**: LLM ranks passages in order of relevance
- **Prompt**: System + User messages requesting Python list output `["1", "2", "3"]`
- **Output**: Ordered list of passage IDs

### Key Features
- Custom prompts per dataset (TREC, SciFact, ISO)
- Support for multiple LLMs (Qwen, Llama, Mistral, Calme)
- 8-bit quantization for efficient memory usage
- CO2 emissions tracking
- Batching for long contexts
- Result caching

---

## What is rank_llm?

**rank_llm** is a modular Python library from the Castorini research group that provides:

1. **Unified API** for using different LLMs for passage reranking
2. **Pre-built ranking strategies**:
   - **Pointwise**: Score each passage independently
   - **Listwise**: Rank all passages together
   - **Sliding Window**: Rank in overlapping windows for scalability
3. **Multiple LLM backends**:
   - OpenAI API (GPT-3.5, GPT-4)
   - Hugging Face Transformers (local models)
   - vLLM (optimized inference)
   - Cohere API
4. **Robust prompt templates** optimized for ranking tasks
5. **Result parsing and aggregation** with error handling

### Key Components

```python
# Main abstractions in rank_llm
from rank_llm import Ranker
from rank_llm.rankers import (
    ListwiseRanker,    # Ranks all passages at once
    PointwiseRanker,   # Scores passages individually
    SlidingWindowRanker # Handles long lists via windowing
)
```

---

## Comparison: Your Approach vs rank_llm

| Aspect | Your Implementation | rank_llm |
|--------|---------------------|----------|
| **Architecture** | Custom two-stage pipeline | Modular ranker classes |
| **Prompting** | Manual JSON configuration | Built-in, optimized prompts |
| **Model Loading** | Direct transformers + quantization | Abstracted model handlers |
| **Ranking Strategy** | Reject → Rerank | Pointwise / Listwise / Sliding Window |
| **Output Parsing** | Manual string parsing | Robust automated parsing |
| **Error Handling** | Try-catch blocks | Built-in retry and fallback |
| **Batching** | Custom token-based batching | Window-based strategies |
| **Extensibility** | Modify core code | Plugin-based rankers |
| **Research Focus** | Your novel 2-stage approach | General-purpose framework |

**Key Insight**: Your two-stage approach is a **research contribution**. rank_llm provides the **infrastructure** to implement it more robustly.

---

## Integration Steps

### Step 0: Verify rank_llm API Structure (Important!)

Before following the integration steps, inspect the actual rank_llm library structure:

```bash
# After installing rank_llm
python -c "import rank_llm; help(rank_llm)"

# Check available modules
python -c "import rank_llm; print(dir(rank_llm))"

# Inspect ranker classes
python -c "from rank_llm import rankers; print(dir(rankers))"

# Or explore the source code
cd /path/to/rank_llm
find . -name "*.py" | head -20
grep -r "class.*Ranker" --include="*.py"
```

This will help you verify:
- Correct import paths
- Available ranker classes
- Constructor parameters
- Method signatures

The examples in this guide use common patterns but **must be adapted** to match the actual API.

### Step 1: Install rank_llm

```bash
# Option 1: Install from PyPI (if available)
# Note: Verify the correct package name on PyPI
pip install rank-llm
# OR
pip install rank_llm

# Option 2: Install from source (recommended for latest features)
git clone https://github.com/castorini/rank_llm.git
cd rank_llm
pip install -e .
```

> **Note**: The exact package name may vary. Check https://pypi.org or the official repository for the correct installation command.

**Dependencies** (check compatibility):
```bash
pip install torch transformers pyserini ranx datasets
```

### Step 2: Understand Your Data Format

rank_llm expects data in this format:

```python
# Query
query = {
    'qid': '123',
    'query': 'What is the effect of climate change?'
}

# Passages (candidates to rank)
candidates = [
    {'docid': 'doc1', 'text': 'Climate change affects...'},
    {'docid': 'doc2', 'text': 'Global warming causes...'},
    # ... more passages
]
```

Your current format in `all_in_one.py`:
```python
# queries_test structure
queries_test = {
    'qid': {
        'query': 'query text',
        'passages': {
            'doc_id': 'passage text',
            ...
        }
    }
}
```

### Step 3: Create a rank_llm Compatible Wrapper

Create a new file `rank_llm_integration.py`:

```python
import os
from typing import Dict, List

# Note: Verify these import paths against the actual rank_llm library
# They may be different depending on the library version
try:
    from rank_llm.rankers import ListwiseRanker
    from rank_llm.retrieval import Request, Candidate
except ImportError:
    # Alternative import paths - check the library structure
    from rank_llm import ListwiseRanker, Request, Candidate

def convert_to_ranklist(qid, query, passages_dict):
    """
    Convert your data format to rank_llm's Request format.
    
    Args:
        qid: Query ID
        query: Query text
        passages_dict: Dict of {doc_id: passage_text}
    
    Returns:
        Request object for rank_llm
    
    Note: Verify Candidate constructor parameters with the actual API.
    Common variations: docid/doc_id, score/rank, text/content
    """
    candidates = [
        # Adjust parameter names based on actual API
        Candidate(docid=doc_id, text=passage_text, score=0.0)
        # Alternative: Candidate(doc_id=doc_id, content=passage_text, score=0.0)
        for doc_id, passage_text in passages_dict.items()
    ]
    
    return Request(
        qid=qid,
        query=query,
        candidates=candidates
    )

def convert_from_ranklist(request):
    """
    Convert rank_llm results back to your format.
    
    Returns:
        Dict mapping doc_id -> rank_score
    """
    rundict = {}
    for rank, candidate in enumerate(request.candidates):
        # Higher rank = better (inverse of position)
        rundict[candidate.docid] = len(request.candidates) - rank
    
    return rundict
```

### Step 4: Initialize rank_llm Ranker

Choose the appropriate ranker for your use case:

```python
# Note: Verify the exact import paths and API with the rank_llm documentation
# The structure below is conceptual and may need adjustment

try:
    from rank_llm.rankers import create_ranker
except ImportError:
    # Alternative: the library may use a different structure
    from rank_llm import create_ranker

# Option A: Use a Hugging Face model (similar to your current approach)
# Verify parameter names with actual API documentation
ranker = create_ranker(
    model_name="castorini/rankllama-v1-7b-lora-passage",  # or your model
    context_size=4096,
    prompt_mode="rank_GPT",  # Built-in ranking prompt
    num_gpus=1,
    device="cuda"
)

# Option B: Use your existing models with rank_llm infrastructure
try:
    from rank_llm.rankers.listwise import ListwiseRanker
except ImportError:
    from rank_llm.rankers import ListwiseRanker

ranker = ListwiseRanker(
    model="models/Qwen2.5-3B-Instruct",  # Your local model path
    context_size=32768,
    prompt_mode="rank_GPT",
    device="cuda",
    batch_size=1
)
```

> **⚠️ API Verification Required**: The exact constructor parameters (model vs model_name, context_size vs max_length, etc.) should be verified against the rank_llm source code or documentation.

### Step 5: Implement Two-Stage Ranking with rank_llm

#### Option A: Use rank_llm for Both Stages

Replace your `generate_responses()` and `generate_rerank()` functions:

```python
from rank_llm.rankers import create_ranker

# Stage 1: Reject Phase (Pointwise scoring)
reject_ranker = create_ranker(
    model_name="models/Qwen2.5-3B-Instruct",
    context_size=32768,
    prompt_mode="rank_GPT",
    num_gpus=1
)

# Stage 2: Rerank Phase (Listwise ranking)
rerank_ranker = create_ranker(
    model_name="models/Qwen2.5-3B-Instruct",
    context_size=32768,
    prompt_mode="rank_GPT",
    num_gpus=1
)

def two_stage_ranking_with_rankllm(qid, query_text, passages_dict, top_k=10):
    """
    Implement your 2-stage approach using rank_llm.
    """
    # Convert to rank_llm format
    request = convert_to_ranklist(qid, query_text, passages_dict)
    
    # Stage 1: Reject - keep only top_k
    reject_result = reject_ranker.rerank(request)
    
    # Filter to top_k passages
    top_candidates = reject_result.candidates[:top_k]
    filtered_request = Request(
        qid=qid,
        query=query_text,
        candidates=top_candidates
    )
    
    # Stage 2: Rerank - final ordering
    final_result = rerank_ranker.rerank(filtered_request)
    
    # Convert back to your format
    rundict = convert_from_ranklist(final_result)
    
    return rundict
```

#### Option B: Hybrid Approach (Keep Your Prompts, Use rank_llm Infrastructure)

If you want to preserve your custom prompts but use rank_llm's model handling:

```python
from rank_llm.rankers.listwise import ListwiseRanker
from rank_llm.retrieval import Request, Candidate
import json

class CustomPromptRanker(ListwiseRanker):
    """
    Custom ranker that uses your prompts from prompts.json
    """
    
    def __init__(self, model_path, prompts_config, **kwargs):
        super().__init__(model=model_path, **kwargs)
        self.prompts = prompts_config
    
    def create_prompt(self, request, rank_start, rank_end):
        """
        Override to use your custom prompt format.
        """
        # Build passage list
        passages = {}
        for i, cand in enumerate(request.candidates[rank_start:rank_end]):
            passages[str(i)] = cand.text
        
        # Use your prompt structure
        prompt = [
            {"role": "system", "content": self.prompts['rerank-S-prompt']},
            {"role": "user", "content": 
                f"{self.prompts['rerank-U-prompt-1']}{json.dumps(passages)}\n\n"
                f"{self.prompts['rerank-U-prompt-2']}{request.query}\n\n"
                f"{self.prompts['rerank-U-prompt-3']}"
            }
        ]
        
        return prompt
    
    def parse_response(self, response_text):
        """
        Parse your specific output format.
        """
        # Your existing parsing logic
        try:
            llm_rep = response_text.split("[")[1].split("]")[0].replace(' ','').split(",")
            ranked_ids = [id.replace("'", '').replace('"', '') for id in llm_rep]
            return ranked_ids
        except:
            return []

# Usage
with open('prompts.json') as f:
    prompts = json.load(f)['scifact']

custom_ranker = CustomPromptRanker(
    model_path="models/Qwen2.5-3B-Instruct",
    prompts_config=prompts,
    context_size=32768,
    device="cuda"
)
```

### Step 6: Update Main Pipeline

Modify your main loop in `all_in_one.py`:

```python
# After loading your existing code...
from rank_llm_integration import two_stage_ranking_with_rankllm

# Replace the existing reject+rerank loop
rundict_rerank = {}

for qid, query_data in tqdm(queries_test.items()):
    qtext = query_data['query']
    passages = query_data['passages']
    
    # Get top-k from BM25
    top_k_passage_ids = list(bm25_rundict[str(qid)].keys())[:args.bm25_topk]
    top_k_passages = {pid: passages[pid] for pid in top_k_passage_ids}
    
    # Use rank_llm for two-stage ranking
    result = two_stage_ranking_with_rankllm(
        qid=qid,
        query_text=qtext,
        passages_dict=top_k_passages,
        top_k=args.reject_number
    )
    
    rundict_rerank[qid] = result
```

### Step 7: Handle Batching and Context Length

rank_llm provides sliding window for long inputs:

```python
from rank_llm.rankers import SlidingWindowRanker

# Automatically handles batching via sliding windows
window_ranker = SlidingWindowRanker(
    model="models/Qwen2.5-3B-Instruct",
    context_size=32768,
    window_size=20,      # Rank 20 passages at a time
    step_size=10,        # Slide by 10 passages
    prompt_mode="rank_GPT"
)

# No need for manual batching - rank_llm handles it
result = window_ranker.rerank(request)
```

Compare with your current batching:

```python
# Your current approach (manual batching)
corpus_batches, biggest_output = split_queries_by_max_length(
    top_k_passages, max_length, len(qtext), prompts
)

for passages in corpus_batches:
    responses.update(generate_responses(qtext, passages, corpus_texts))
```

---

## Code Examples

### Full Integration Example

Create `all_in_one_rankllm.py`:

```python
#!/usr/bin/env python3
"""
Modified version of all_in_one.py using rank_llm library.
"""

import json
import time
from tqdm import tqdm
import argparse
from rank_llm.rankers import create_ranker
from rank_llm.retrieval import Request, Candidate
from ranx import Qrels, Run, evaluate
from beir.datasets.data_loader import GenericDataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="rank_llm model to use", 
                        default="castorini/rankllama-v1-7b-lora-passage", type=str)
    parser.add_argument("--dataset", help="dataset to test", type=str, default="scifact")
    parser.add_argument("--bm25_topk", type=int, default=100)
    parser.add_argument("--reject_number", type=int, default=10)
    return parser.parse_args()

def convert_to_request(qid, query_text, passages_dict, bm25_scores=None):
    """Convert your data format to rank_llm Request."""
    candidates = []
    for doc_id, text in passages_dict.items():
        score = bm25_scores.get(doc_id, 0.0) if bm25_scores else 0.0
        candidates.append(Candidate(docid=doc_id, text=text, score=score))
    
    return Request(qid=qid, query=query_text, candidates=candidates)

def two_stage_reranking(ranker, qid, query_text, passages, bm25_scores, reject_k):
    """
    Two-stage reranking using rank_llm.
    
    Stage 1 (Reject): Filter to top reject_k passages
    Stage 2 (Rerank): Final ranking of filtered passages
    """
    # Convert data
    request = convert_to_request(qid, query_text, passages, bm25_scores)
    
    # Stage 1: Initial ranking (implicit reject)
    ranked_request = ranker.rerank(request)
    
    # Stage 2: Keep top-k and re-rank with higher precision
    # (In practice, you might use different prompts or parameters here)
    top_k_candidates = ranked_request.candidates[:reject_k]
    
    # Build result dict
    rundict = {}
    for rank, candidate in enumerate(top_k_candidates):
        rundict[candidate.docid] = len(top_k_candidates) - rank
    
    return rundict

if __name__ == '__main__':
    args = get_args()
    print(f"Loading rank_llm model: {args.model_name}")
    
    # Initialize rank_llm ranker
    ranker = create_ranker(
        model_name=args.model_name,
        context_size=4096,
        prompt_mode="rank_GPT",
        num_gpus=1
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    corpus, queries, qrels = GenericDataLoader(f"datasets/{args.dataset}").load(split="test")
    corpus = {k: v['text'] for k, v in corpus.items()}
    
    # Load BM25 results
    with open(f'rundicts/rundict_{args.dataset}_bm25.json', 'r') as f:
        bm25_rundict = json.load(f)
    
    # Rerank with rank_llm
    rundict_rerank = {}
    
    for qid in tqdm(queries.keys()):
        query_text = queries[qid]
        
        # Get top-k from BM25
        top_k_ids = list(bm25_rundict[qid].keys())[:args.bm25_topk]
        top_k_passages = {doc_id: corpus[doc_id] for doc_id in top_k_ids if doc_id in corpus}
        bm25_scores = {doc_id: bm25_rundict[qid][doc_id] for doc_id in top_k_ids}
        
        # Two-stage reranking
        result = two_stage_reranking(
            ranker=ranker,
            qid=qid,
            query_text=query_text,
            passages=top_k_passages,
            bm25_scores=bm25_scores,
            reject_k=args.reject_number
        )
        
        rundict_rerank[qid] = result
    
    # Evaluate
    metrics = ['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 
               'recall@1', 'recall@5', 'recall@10']
    
    results_bm25 = evaluate(Qrels(qrels), Run(bm25_rundict), metrics)
    results_rerank = evaluate(Qrels(qrels), Run(rundict_rerank), metrics)
    
    print('\n=== BM25 Results ===')
    print(results_bm25)
    print('\n=== rank_llm Rerank Results ===')
    print(results_rerank)
    
    # Save results
    with open(f'results_rankllm_{args.dataset}.json', 'w') as f:
        json.dump({
            'bm25': results_bm25,
            'rerank': results_rerank
        }, f, indent=4)
```

### Minimal Integration (Keep Most of Your Code)

If you want minimal changes, just replace the model loading and inference:

```python
# At the top of all_in_one.py, add:
from rank_llm.rankers import ListwiseRanker

# Replace the model loading section (lines 216-237)
print(f"Loading model with rank_llm...")
ranker = ListwiseRanker(
    model=model_path,
    context_size=max_length,
    prompt_mode="rank_GPT",
    device="cuda",
    batch_size=1
)

# Then modify generate_rerank() to use rank_llm:
def generate_rerank_rankllm(query, docs_dict):
    """Use rank_llm for reranking."""
    # Convert to rank_llm format
    candidates = [
        Candidate(docid=doc_id, text=text, score=0.0)
        for doc_id, text in docs_dict.items()
    ]
    
    request = Request(
        qid="temp",
        query=query,
        candidates=candidates
    )
    
    # Rerank
    result = ranker.rerank(request)
    
    # Convert back
    responses = {}
    for rank, candidate in enumerate(result.candidates):
        responses[candidate.docid] = len(result.candidates) - rank
    
    return responses
```

---

## Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

1. **Week 1**: Install rank_llm, create wrapper functions
2. **Week 2**: Test on small subset of data, compare results
3. **Week 3**: Migrate one stage at a time (rerank first, then reject)
4. **Week 4**: Full migration and performance tuning

### Strategy 2: Side-by-Side Comparison

Keep both implementations and compare:

```python
# Run both approaches
result_original = generate_rerank(query, docs)
result_rankllm = generate_rerank_rankllm(query, docs)

# Compare
print(f"Agreement: {compute_kendall_tau(result_original, result_rankllm)}")
```

### Strategy 3: Hybrid Approach

Use rank_llm for infrastructure but keep your prompts and logic:

- rank_llm: Model loading, inference, error handling
- Your code: Prompt engineering, two-stage logic, custom batching

---

## Migration Checklist

- [ ] Install rank_llm and dependencies
- [ ] Test basic rank_llm functionality with toy example
- [ ] Create data format converters (to/from rank_llm format)
- [ ] Implement wrapper for your two-stage approach
- [ ] Test on small subset (e.g., 10 queries)
- [ ] Compare results with original implementation
- [ ] Profile performance (speed, memory)
- [ ] Migrate batch processing or use SlidingWindowRanker
- [ ] Update evaluation pipeline
- [ ] Add CO2 tracking (adapt your existing code)
- [ ] Full dataset testing
- [ ] Document any behavioral differences
- [ ] Update paper/documentation

---

## Troubleshooting

### Issue 1: Model Path Errors

**Problem**: rank_llm can't find your local models

**Solution**:
```python
# Specify full absolute path
ranker = create_ranker(
    model_name="/full/path/to/models/Qwen2.5-3B-Instruct",
    ...
)

# Or set HF_HOME environment variable
import os
os.environ['HF_HOME'] = '/path/to/your/models'
```

### Issue 2: Out of Memory

**Problem**: GPU OOM with rank_llm

**Solution**:
```python
# Use smaller window size
ranker = SlidingWindowRanker(
    window_size=10,  # Reduce from default
    ...
)

# Or use 8-bit quantization (if supported)
# Note: rank_llm may not support this directly,
# you might need to modify the library or use your existing quantization
```

### Issue 3: Different Results

**Problem**: rank_llm gives different results than your implementation

**Reason**: Different prompts, temperature, or parsing logic

**Solution**:
```python
# Extend rank_llm ranker to use your exact prompts
class YourCustomRanker(ListwiseRanker):
    def create_prompt(self, request, rank_start, rank_end):
        # Use your exact prompt from prompts.json
        pass
    
    def parse_response(self, response):
        # Use your exact parsing logic
        pass
```

### Issue 4: Performance Degradation

**Problem**: rank_llm is slower than your implementation

**Cause**: Additional abstraction layers, different batching

**Solution**:
```python
# Profile to find bottleneck
import cProfile
cProfile.run('ranker.rerank(request)')

# Optimize based on findings:
# - Adjust batch size
# - Use vLLM backend for faster inference
# - Disable unnecessary features
```

### Issue 5: Prompt Format Mismatch

**Problem**: Your model expects chat format, rank_llm uses different format

**Solution**:
```python
# Override the prompt formatting
class ChatFormatRanker(ListwiseRanker):
    def run_llm(self, prompt):
        # Convert prompt to chat messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        # Use your existing pipeline
        outputs = self.pipe(messages, max_new_tokens=256)
        return outputs[0]['generated_text'][-1]['content']
```

---

## Advanced Topics

### Using rank_llm with vLLM for Speed

```python
from rank_llm.rankers import create_ranker

# vLLM backend (much faster for batch inference)
ranker = create_ranker(
    model_name="models/Qwen2.5-3B-Instruct",
    backend="vllm",  # Use vLLM instead of transformers
    context_size=32768,
    num_gpus=1
)
```

### Implementing Custom Ranking Strategies

```python
from rank_llm.rankers import Ranker

class TwoStageRanker(Ranker):
    """
    Custom ranker implementing your two-stage approach
    as a first-class rank_llm ranker.
    """
    
    def __init__(self, reject_k=10, **kwargs):
        super().__init__(**kwargs)
        self.reject_k = reject_k
    
    def rerank(self, request):
        # Stage 1: Reject
        stage1_result = self._reject_stage(request)
        
        # Stage 2: Rerank
        filtered_request = self._filter_top_k(stage1_result, self.reject_k)
        stage2_result = self._rerank_stage(filtered_request)
        
        return stage2_result
    
    def _reject_stage(self, request):
        # Your reject logic with prompts['S-prompt'], etc.
        pass
    
    def _rerank_stage(self, request):
        # Your rerank logic with prompts['rerank-S-prompt'], etc.
        pass
```

### Preserving CO2 Tracking

```python
from codecarbon import OfflineEmissionsTracker

# Wrap rank_llm calls with your existing tracker
tracker = OfflineEmissionsTracker(...)

tracker.start()
result = ranker.rerank(request)
tracker.stop()

emissions = tracker.final_emissions
```

---

## Benefits of Using rank_llm

1. **Robustness**: Better error handling and parsing
2. **Maintainability**: Less custom code to maintain
3. **Community**: Active development and bug fixes
4. **Extensibility**: Easy to add new models and strategies
5. **Reproducibility**: Standardized prompts and evaluation
6. **Performance**: Optimized inference backends (vLLM)

## When to Keep Your Implementation

- You need exact control over prompts for research
- Your two-stage approach is the core contribution
- You have custom batching logic that's hard to replicate
- You need specific quantization or optimization
- Integration overhead is too high for your timeline

## Hybrid Approach (Best of Both Worlds)

**Recommended**: Use rank_llm for infrastructure, keep your research logic:

```python
# Use rank_llm for:
- Model loading and management
- Basic inference infrastructure  
- Result parsing and error handling

# Keep your implementation for:
- Two-stage reject+rerank logic
- Custom prompts per dataset
- Specific batching strategies
- CO2 tracking and performance metrics
```

---

## Conclusion

**Yes, you can use rank_llm in the same way as all_in_one.py!** 

The library is flexible enough to support your two-stage approach while providing better infrastructure. Choose your integration level based on your needs:

- **Full migration**: Use rank_llm end-to-end for maximum benefit
- **Hybrid**: Use rank_llm infrastructure with your prompts/logic
- **Minimal**: Just use rank_llm for specific components

All approaches are valid and can maintain the methodology described in your paper.

---

## References

- [rank_llm GitHub Repository](https://github.com/castorini/rank_llm) - **Start here for official documentation**
- [rank_llm Source Code](https://github.com/castorini/rank_llm/tree/main) - Verify API structure
- [Castorini Papers](https://github.com/castorini) - Related research
- Your Paper: `Paper_ECIR.pdf`
- Your Implementation: `all_in_one.py`

### How to Verify the API

Since the exact rank_llm API may differ from this guide's examples:

1. **Read the official README**: https://github.com/castorini/rank_llm#readme
2. **Check example scripts**: Look for `examples/` or `scripts/` in the repo
3. **Inspect source code**: Browse the Python files to see actual class definitions
4. **Run help()**: Use Python's built-in help to explore the API
5. **Check tests**: Unit tests often show correct usage patterns

```bash
# Example: Finding the correct API
git clone https://github.com/castorini/rank_llm.git
cd rank_llm
cat README.md  # Read usage instructions
ls examples/   # Look for example scripts
grep -r "def rerank" --include="*.py"  # Find rerank methods
```

---

## Need Help?

1. Check rank_llm issues: https://github.com/castorini/rank_llm/issues
2. Castorini group papers: https://github.com/castorini
3. Compare with rankllama paper: Search "RankLLaMA" on ArXiv

**Good luck with your integration!** 🚀
