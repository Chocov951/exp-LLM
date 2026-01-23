#!/usr/bin/env python3
"""
Full pipeline for ranking evaluation using rank_llm library with two-stage strategy.

This script implements:
1. Full ranking evaluation pipeline using rank_llm library
2. CodeCarbon tracking for ranking cost
3. Support for multiple datasets from rank_llm
4. Top100 docs from pyserini BM25 index for each dataset
5. Parametrable two-stage ranking strategy:
   - Filter step: custom LLM & prompt, pointwise, or BERT index
   - Ranking step: custom LLM & prompt or rank_llm existing methods
6. Export all metrics (NDCG, MRR, Recall, CodeCarbon) to JSON
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModel
from codecarbon import OfflineEmissionsTracker
from beir.datasets.data_loader import GenericDataLoader
from pyserini.search.lucene import LuceneSearcher
from ranx import Qrels, Run, evaluate
import csv


# Try to import rank_llm components - will handle gracefully if not installed
RANK_LLM_AVAILABLE = False
try:
    from rank_llm.data import Request, Query, Candidate, Result
    from rank_llm.rerank.listwise import (
        SafeOpenai,
        VicunaReranker,
        ZephyrReranker,
        RankListwiseOSLLM,
    )
    from rank_llm.rerank.pointwise import PointwiseRankLLM
    from rank_llm.rerank.rankllm import PromptMode
    RANK_LLM_AVAILABLE = True
except ImportError:
    print("Warning: rank_llm not fully installed. Will use fallback implementations.")
    # Define minimal classes for compatibility
    class Query:
        def __init__(self, text, qid):
            self.text = text
            self.qid = qid
    
    class Candidate:
        def __init__(self, docid, score=0.0, doc=None):
            self.docid = docid
            self.score = score
            self.doc = doc if doc is not None else {}
    
    class Request:
        def __init__(self, query, candidates):
            self.query = query
            self.candidates = candidates
    
    class Result:
        def __init__(self, query, candidates):
            self.query = query
            self.candidates = candidates


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RankLLM evaluation pipeline with two-stage strategy")
    
    # Dataset and model configuration
    parser.add_argument("--dataset", type=str, default="scifact", 
                       help="Dataset to test (scifact, trec-covid, fiqa, etc.)")
    parser.add_argument("--model_name", type=str, default="qwen3",
                       choices=["qwen3", "qwen14", "qwen32", "qwen72", "calme", "llama", "mistral"],
                       help="Model to use for custom LLM ranking")
    parser.add_argument("--bm25_topk", type=int, default=100,
                       help="Top-k documents from BM25 to rerank")
    
    # Two-stage strategy configuration
    parser.add_argument("--filter_method", type=str, default="custom_llm",
                       choices=["custom_llm", "pointwise", "bert_index"],
                       help="Method for filter stage")
    parser.add_argument("--filter_topk", type=int, default=10,
                       help="Number of passages to keep after filter stage")
    parser.add_argument("--ranking_method", type=str, default="custom_llm",
                       choices=["custom_llm", "rankllm_listwise", "rankllm_pointwise"],
                       help="Method for ranking stage")
    parser.add_argument("--stage", type=str, default="both",
                       choices=["filter", "rerank", "both"],
                       help="Select which stage(s) to run: filter only, rerank only, or both")
    parser.add_argument("--filtered_rundict", type=str, default=None,
                       help="Path to precomputed filtered rundict JSON for reranking stage")
    parser.add_argument("--filtered_output", type=str, default=None,
                       help="Path to save filtered rundict JSON when running filter stage")
    
    # RankLLM specific options
    parser.add_argument("--rankllm_model", type=str, default="castorini/rank_zephyr_7b_v1_full",
                       help="RankLLM model name if using rankllm methods")
    parser.add_argument("--bert_model", type=str, default="BAAI/bge-m3",
                       help="BERT model for BERT index filtering")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="rankllm_results",
                       help="Directory to save results")
    parser.add_argument("--use_cache", action="store_true",
                       help="Use cached results if available")
    
    return parser.parse_args()


def create_model_config(model_name: str) -> Tuple[str, int, Optional[BitsAndBytesConfig]]:
    """Create model configuration based on model name."""
    model_configs = {
        'qwen3': ('models/Qwen2.5-3B-Instruct', 32768, True),
        'qwen14': ('models/Qwen2.5-14B-Instruct', 32768, True),
        'qwen32': ('models/Qwen2.5-32B-Instruct', 32768, True),
        'qwen72': ('models/Qwen2.5-72B-Instruct', 32768, True),
        'calme': ('models/calme-3.1-instruct-78b', 32768, True),
        'llama': ('models/llama-3.3-70B-Instruct', 32768, True),
        'mistral': ('models/Mistral-Small-3.1-24B-Instruct-2503', 32768, True),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Model {model_name} not supported")
    
    model_path, max_length, use_quant = model_configs[model_name]
    
    quantization_config = None
    if use_quant:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    return model_path, max_length, quantization_config


class FilterStage:
    """Filter stage implementation separated from ranking."""

    def __init__(self, args, prompts: Dict[str, str]):
        self.args = args
        self.prompts = prompts
        self.filter_method = args.filter_method
        self.filter_topk = args.filter_topk

        self.max_length = None
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.bert_tokenizer = None
        self.bert_model = None

        self.filter_time = 0
        self.filter_count = 0

        self._init_models()

    def _init_models(self):
        """Initialize filter models based on configuration."""
        if self.filter_method == "custom_llm":
            self._init_custom_llm()
        elif self.filter_method == "bert_index":
            self._init_bert_model()
        elif self.filter_method == "pointwise":
            if RANK_LLM_AVAILABLE:
                print("Pointwise initialization would go here")
            else:
                print("Warning: Pointwise requires rank_llm library")

    def _init_custom_llm(self):
        """Initialize custom LLM model for filtering."""
        print(f"Loading custom LLM for filter stage: {self.args.model_name}")
        model_path, max_length, quantization_config = create_model_config(self.args.model_name)

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            max_length=max_length,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        print("Custom LLM (filter) loaded successfully")

    def _init_bert_model(self):
        """Initialize BERT model for filtering."""
        print(f"Loading BERT model: {self.args.bert_model}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)
        self.bert_model = AutoModel.from_pretrained(self.args.bert_model).to('cuda')
        print("BERT model loaded successfully")

    def filter(self, query: str, passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """Run the filter stage and return a relevance score dict."""
        start_time = time.time()

        if self.filter_method == "custom_llm":
            result = self._filter_custom_llm(query, passages)
        elif self.filter_method == "bert_index":
            result = self._filter_bert(query, passages)
        elif self.filter_method == "pointwise":
            result = self._filter_pointwise(query, passages)
        else:
            raise ValueError(f"Unknown filter method: {self.filter_method}")

        self.filter_time += time.time() - start_time
        self.filter_count += 1
        return result

    def _filter_custom_llm(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Filter using custom LLM with prompts."""
        plist_s = {str(i): passage for i, passage in enumerate(passages.values())}
        passage_ids = list(passages.keys())

        messages = [
            {"role": "system", "content": self.prompts['S-prompt']},
            {"role": "user", "content":
                f"{self.prompts['U-prompt-1']}{json.dumps(plist_s)}\n\n"
                f"{self.prompts['U-prompt-2']}{query}\n\n"
                f"{self.prompts['U-prompt-3']}"
            }
        ]

        try:
            outputs = self.pipe(messages, max_new_tokens=10*self.filter_topk+10,
                               num_return_sequences=1, do_sample=False)
            llm_rep = outputs[0]['generated_text'][-1]['content']

            llm_rep = llm_rep.split("{")[1]
            if '}' in llm_rep:
                llm_rep = llm_rep.split("}")[0]
            else:
                llm_rep = llm_rep.rsplit(",", 1)[0]
            llm_rep = '{' + llm_rep + '}'
            llm_rep = llm_rep.replace("'", '"')
            llm_rep = json.loads(llm_rep)

            result = {}
            for idx_str, score in llm_rep.items():
                if idx_str in plist_s:
                    original_idx = int(idx_str)
                    if original_idx < len(passage_ids):
                        result[passage_ids[original_idx]] = float(score)

            return result
        except Exception as e:
            print(f"Error in custom LLM filtering: {e}")
            return {}

    def _filter_bert(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Filter using BERT similarity."""
        query_inputs = self.bert_tokenizer(query, return_tensors="pt",
                                          truncation=True, padding=True).to('cuda')
        with torch.no_grad():
            query_outputs = self.bert_model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state[:, 0, :].cpu()

        similarities = {}
        for doc_id, passage_text in passages.items():
            passage_inputs = self.bert_tokenizer(passage_text, return_tensors="pt",
                                                truncation=True, padding=True).to('cuda')
            with torch.no_grad():
                passage_outputs = self.bert_model(**passage_inputs)
                passage_embedding = passage_outputs.last_hidden_state[:, 0, :].cpu()

            similarity = torch.nn.functional.cosine_similarity(
                query_embedding, passage_embedding
            ).item()
            similarities[doc_id] = similarity

        return similarities

    def _filter_pointwise(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Filter using Pointwise from rank_llm."""
        print("Pointwise filtering not yet implemented, falling back to BERT")
        return self._filter_bert(query, passages)

    def export_shared_llm(self) -> Optional[Dict[str, object]]:
        """Expose custom LLM components for reuse by ranking stage."""
        if self.filter_method != "custom_llm":
            return None
        return {
            "pipe": self.pipe,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_length": self.max_length,
        }


class RankingStage:
    """Ranking stage implementation separated from filtering."""

    def __init__(self, args, prompts: Dict[str, str], shared_custom_llm: Optional[Dict[str, object]] = None):
        self.args = args
        self.prompts = prompts
        self.ranking_method = args.ranking_method

        self.max_length = None
        self.tokenizer = None
        self.model = None
        self.pipe = None
        
        # RankLLM rerankers
        self.rankllm_reranker = None

        self.ranking_time = 0
        self.ranking_count = 0

        self._init_models(shared_custom_llm)

    def _init_models(self, shared_custom_llm: Optional[Dict[str, object]]):
        """Initialize ranking models based on configuration."""
        if self.ranking_method == "custom_llm":
            if shared_custom_llm and shared_custom_llm.get("pipe"):
                print("Reusing custom LLM from filter stage for ranking")
                self.pipe = shared_custom_llm.get("pipe")
                self.model = shared_custom_llm.get("model")
                self.tokenizer = shared_custom_llm.get("tokenizer")
                self.max_length = shared_custom_llm.get("max_length")
            else:
                self._init_custom_llm()
        elif self.ranking_method == "rankllm_listwise":
            self._init_rankllm_listwise()
        elif self.ranking_method == "rankllm_pointwise":
            self._init_rankllm_pointwise()

    def _init_custom_llm(self):
        """Initialize custom LLM model for ranking."""
        print(f"Loading custom LLM for ranking stage: {self.args.model_name}")
        model_path, max_length, quantization_config = create_model_config(self.args.model_name)

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            max_length=max_length,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        print("Custom LLM (ranking) loaded successfully")
    
    def _init_rankllm_listwise(self):
        """Initialize RankLLM listwise reranker."""
        if not RANK_LLM_AVAILABLE:
            print("Warning: rank_llm not available, cannot initialize listwise reranker")
            return
        
        print(f"Loading RankLLM listwise reranker: {self.args.rankllm_model}")
        try:
            # Use ZephyrReranker by default, which is a popular choice
            # Users can modify this to use VicunaReranker or other models
            if "zephyr" in self.args.rankllm_model.lower():
                self.rankllm_reranker = ZephyrReranker(
                    model_path=self.args.rankllm_model,
                    context_size=4096,
                    num_gpus=1,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    window_size=20,
                    stride=10,
                )
            elif "vicuna" in self.args.rankllm_model.lower():
                self.rankllm_reranker = VicunaReranker(
                    model_path=self.args.rankllm_model,
                    context_size=4096,
                    num_gpus=1,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    window_size=20,
                    stride=10,
                )
            else:
                # Generic listwise reranker
                self.rankllm_reranker = RankListwiseOSLLM(
                    model=self.args.rankllm_model,
                    context_size=4096,
                    num_gpus=1,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    window_size=20,
                    stride=10,
                )
            print("RankLLM listwise reranker loaded successfully")
        except Exception as e:
            print(f"Error initializing RankLLM listwise reranker: {e}")
            print("Falling back to custom LLM")
            self.rankllm_reranker = None
    
    def _init_rankllm_pointwise(self):
        """Initialize RankLLM pointwise reranker."""
        if not RANK_LLM_AVAILABLE:
            print("Warning: rank_llm not available, cannot initialize pointwise reranker")
            return
        
        print(f"Loading RankLLM pointwise reranker: {self.args.rankllm_model}")
        try:
            # PointwiseRankLLM for pointwise ranking
            self.rankllm_reranker = PointwiseRankLLM(
                model=self.args.rankllm_model,
                context_size=4096,
                num_gpus=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print("RankLLM pointwise reranker loaded successfully")
        except Exception as e:
            print(f"Error initializing RankLLM pointwise reranker: {e}")
            print("Falling back to custom LLM")
            self.rankllm_reranker = None

    def rank(self, query: str, filtered_passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """Run the ranking stage and return rank scores."""
        start_time = time.time()

        if self.ranking_method == "custom_llm":
            result = self._rank_custom_llm(query, filtered_passages)
        elif self.ranking_method == "rankllm_listwise":
            result = self._rank_rankllm_listwise(query, filtered_passages)
        elif self.ranking_method == "rankllm_pointwise":
            result = self._rank_rankllm_pointwise(query, filtered_passages)
        else:
            raise ValueError(f"Unknown ranking method: {self.ranking_method}")

        self.ranking_time += time.time() - start_time
        self.ranking_count += 1
        return result

    def _rank_custom_llm(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Rank using custom LLM with prompts."""
        messages = [
            {"role": "system", "content": self.prompts['rerank-S-prompt']},
            {"role": "user", "content":
                f"{self.prompts['rerank-U-prompt-1']}{json.dumps(passages)}\n\n"
                f"{self.prompts['rerank-U-prompt-2']}{query}\n\n"
                f"{self.prompts['rerank-U-prompt-3']}"
            }
        ]

        try:
            outputs = self.pipe(messages, max_new_tokens=10*len(passages)+10,
                               num_return_sequences=1, do_sample=False)
            llm_rep = outputs[0]['generated_text'][-1]['content']

            llm_rep = llm_rep.split("[")[1].split("]")[0].replace(' ', '').split(",")

            result = {}
            for rank, doc_id in enumerate(llm_rep):
                doc_id = doc_id.replace("'", '').replace('"', '')
                if doc_id in passages:
                    result[doc_id] = len(llm_rep) - rank

            return result
        except Exception as e:
            print(f"Error in custom LLM ranking: {e}")
            return {}

    def _rank_rankllm_listwise(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Rank using rank_llm listwise ranker."""
        if not RANK_LLM_AVAILABLE or self.rankllm_reranker is None:
            print("Warning: rank_llm not available, falling back to custom LLM")
            return self._rank_custom_llm(query, passages)

        try:
            # Convert to RankLLM format
            query_obj = Query(text=query, qid="temp")
            candidates = []
            for doc_id, text in passages.items():
                candidates.append(Candidate(
                    docid=doc_id,
                    score=0.0,
                    doc={"text": text, "title": ""}
                ))
            
            request = Request(query=query_obj, candidates=candidates)
            
            # Rerank using RankLLM
            result = self.rankllm_reranker.rerank(
                request=request,
                rank_start=0,
                rank_end=len(passages)
            )
            
            # Convert back to our format
            reranked_dict = {}
            for rank, candidate in enumerate(result.candidates):
                # Higher score for higher rank (inverse of position)
                reranked_dict[str(candidate.docid)] = len(result.candidates) - rank
            
            return reranked_dict
        except Exception as e:
            print(f"Error in RankLLM listwise ranking: {e}")
            print("Falling back to custom LLM")
            return self._rank_custom_llm(query, passages)

    def _rank_rankllm_pointwise(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Rank using rank_llm pointwise ranker."""
        if not RANK_LLM_AVAILABLE or self.rankllm_reranker is None:
            print("Warning: rank_llm not available, falling back to custom LLM")
            return self._rank_custom_llm(query, passages)

        try:
            # Convert to RankLLM format
            query_obj = Query(text=query, qid="temp")
            candidates = []
            for doc_id, text in passages.items():
                candidates.append(Candidate(
                    docid=doc_id,
                    score=0.0,
                    doc={"text": text, "title": ""}
                ))
            
            request = Request(query=query_obj, candidates=candidates)
            
            # Rerank using RankLLM
            result = self.rankllm_reranker.rerank(
                request=request,
                rank_start=0,
                rank_end=len(passages)
            )
            
            # Convert back to our format
            reranked_dict = {}
            for rank, candidate in enumerate(result.candidates):
                # Higher score for higher rank (inverse of position)
                reranked_dict[str(candidate.docid)] = len(result.candidates) - rank
            
            return reranked_dict
        except Exception as e:
            print(f"Error in RankLLM pointwise ranking: {e}")
            print("Falling back to custom LLM")
            return self._rank_custom_llm(query, passages)


class TwoStageRanker:
    """
    Combined two-stage ranker for compatibility.
    Combines FilterStage and RankingStage into a single interface.
    """
    def __init__(self, args, prompts: Dict[str, str]):
        self.args = args
        self.prompts = prompts
        self.filter_stage = FilterStage(args, prompts)
        self.ranking_stage = RankingStage(args, prompts, self.filter_stage.export_shared_llm())
    
    def two_stage_rerank(self, query: str, passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """
        Run complete two-stage reranking.
        
        Args:
            query: Query text
            passages: Dict of {doc_id: passage_text}
            qid: Query ID
            
        Returns:
            Dict of {doc_id: final_score}
        """
        # Stage 1: Filter
        filter_scores = self.filter_stage.filter(query, passages, qid)
        if not filter_scores:
            return {}
        
        # Get top-k from filter stage
        sorted_filtered = sorted(filter_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_passages = {pid: passages[pid] for pid, _ in sorted_filtered[:self.args.filter_topk] if pid in passages}
        
        if not top_k_passages:
            return {}
        
        # Stage 2: Rank
        final_scores = self.ranking_stage.rank(query, top_k_passages, qid)
        return final_scores


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


def save_results(args, metrics_dict: Dict, codecarbon_metrics: Dict, output_dir: str):
    """Save all metrics to JSON file with descriptive filename."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with hyperparameters
    filename = (
        f"{args.dataset}_"
        f"{args.model_name}_"
        f"filter-{args.filter_method}_"
        f"rank-{args.ranking_method}_"
        f"topk{args.filter_topk}_"
        f"results.json"
    )
    
    filepath = os.path.join(output_dir, filename)
    
    # Combine all results
    full_results = {
        'config': {
            'dataset': args.dataset,
            'model_name': args.model_name,
            'filter_method': args.filter_method,
            'ranking_method': args.ranking_method,
            'bm25_topk': args.bm25_topk,
            'filter_topk': args.filter_topk,
            'rankllm_model': args.rankllm_model if hasattr(args, 'rankllm_model') else None,
            'bert_model': args.bert_model if hasattr(args, 'bert_model') else None,
        },
        'metrics': metrics_dict,
        'codecarbon': codecarbon_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=4)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def save_filtered_rundict(filtered_rundict: Dict, output_path: str):
    """Persist filtered rundict to disk for later reranking."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_rundict, f, indent=2)
    print(f"Filtered rundict saved to: {output_path}")


def main():
    """Main execution function."""
    args = get_args()
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
        corpus, queries, qrels = GenericDataLoader(f"datasets/{args.dataset}").load(split="test")
        corpus = {k: v['text'] if isinstance(v, dict) else v for k, v in corpus.items()}
        print(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels")
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

    filter_emissions_file = (
        f"emissions_filter_{args.dataset}_{args.filter_method}_"
        f"topk{args.filter_topk}.csv"
    )
    ranking_emissions_file = (
        f"emissions_ranking_{args.dataset}_{args.ranking_method}_"
        f"topk{args.filter_topk}.csv"
    )

    tracker_filter = None
    tracker_ranking = None

    filtered_rundict = None
    filter_engine = None

    queries_to_process = [qid for qid in qrels.keys() if qid in queries]
    print(f"Processing {len(queries_to_process)} queries")

    filtered_output_path = args.filtered_output or os.path.join(
        args.output_dir,
        f"filtered_{args.dataset}_{args.filter_method}_topk{args.filter_topk}.json"
    )

    if args.stage in ("filter", "both"):
        print("\nRunning filter stage...")
        filter_engine = FilterStage(args, prompts)
        tracker_filter = OfflineEmissionsTracker(
            country_iso_code="FRA",
            tracking_mode="process",
            on_csv_write="append",
            project_name="RankLLM-Filter",
            log_level="warning",
            output_dir=emissions_dir,
            output_file=filter_emissions_file
        )
        tracker_filter.start()

        filtered_rundict = {}
        for qid in tqdm(queries_to_process, desc="Filtering queries"):
            query_text = queries[qid]

            if str(qid) not in bm25_rundict and qid not in bm25_rundict:
                print(f"Warning: Query {qid} not in BM25 results")
                continue

            bm25_qid = str(qid) if str(qid) in bm25_rundict else qid
            top_k_passage_ids = list(bm25_rundict[bm25_qid].keys())[:args.bm25_topk]

            passages = {}
            for pid in top_k_passage_ids:
                if pid in corpus:
                    passages[pid] = corpus[pid]
                elif str(pid) in corpus:
                    passages[str(pid)] = corpus[str(pid)]

            if not passages:
                print(f"Warning: No passages found for query {qid}")
                continue

            try:
                filter_scores = filter_engine.filter(query_text, passages, str(qid))
                if not filter_scores:
                    continue
                sorted_passages = sorted(filter_scores.items(), key=lambda x: x[1], reverse=True)
                filtered_rundict[str(qid)] = {pid: score for pid, score in sorted_passages[:args.filter_topk]}
            except Exception as e:
                print(f"Error filtering query {qid}: {e}")
                continue

        tracker_filter.stop()
        save_filtered_rundict(filtered_rundict, filtered_output_path)

        if args.stage == "filter":
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
            codecarbon_metrics['timing'] = {
                'filter_avg_time_s': filter_engine.filter_time / max(filter_engine.filter_count, 1),
                'filter_total_time_s': filter_engine.filter_time,
            }

            print("\nFilter stage complete. Reranking skipped (stage=filter).")
            print(json.dumps(codecarbon_metrics, indent=2))
            return

    if args.stage == "rerank":
        if not args.filtered_rundict:
            print("Error: --filtered_rundict is required for rerank-only stage")
            return
        if not os.path.exists(args.filtered_rundict):
            print(f"Error: Provided filtered rundict not found at {args.filtered_rundict}")
            return
        with open(args.filtered_rundict, 'r', encoding='utf-8') as f:
            filtered_rundict = json.load(f)

    if not filtered_rundict:
        print("Error: No filtered rundict available for ranking stage")
        return

    print("\nRunning ranking stage...")
    ranking_engine = RankingStage(args, prompts, shared_custom_llm=filter_engine.export_shared_llm() if filter_engine else None)
    tracker_ranking = OfflineEmissionsTracker(
        country_iso_code="FRA",
        tracking_mode="process",
        on_csv_write="append",
        project_name="RankLLM-Ranking",
        log_level="warning",
        output_dir=emissions_dir,
        output_file=ranking_emissions_file
    )
    tracker_ranking.start()

    rundict_rerank = {}
    for qid in tqdm(queries_to_process, desc="Reranking queries"):
        query_text = queries[qid]
        filtered_scores = filtered_rundict.get(str(qid)) or filtered_rundict.get(qid)
        if not filtered_scores:
            print(f"Warning: No filtered docs for query {qid}")
            continue

        sorted_filtered = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        filtered_passages = {}
        for pid, _ in sorted_filtered:
            if pid in corpus:
                filtered_passages[pid] = corpus[pid]
            elif str(pid) in corpus:
                filtered_passages[str(pid)] = corpus[str(pid)]

        if not filtered_passages:
            print(f"Warning: No passages found for reranking query {qid}")
            continue

        try:
            final_scores = ranking_engine.rank(query_text, filtered_passages, str(qid))
            rundict_rerank[str(qid)] = final_scores
        except Exception as e:
            print(f"Error reranking query {qid}: {e}")
            continue

    tracker_ranking.stop()

    print("\nEvaluating results...")

    bm25_metrics = evaluate_metrics(qrels, bm25_rundict) if bm25_rundict else {}
    if bm25_metrics:
        print("BM25 baseline metrics:")
        print(json.dumps(bm25_metrics, indent=2))

    rerank_metrics = evaluate_metrics(qrels, rundict_rerank)
    print("\nTwo-stage reranking metrics:")
    print(json.dumps(rerank_metrics, indent=2))

    codecarbon_metrics = {}

    if tracker_filter:
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

    timing_metrics = {}
    if filter_engine:
        timing_metrics['filter_avg_time_s'] = filter_engine.filter_time / max(filter_engine.filter_count, 1)
        timing_metrics['filter_total_time_s'] = filter_engine.filter_time
    else:
        timing_metrics['filter_avg_time_s'] = 0
        timing_metrics['filter_total_time_s'] = 0

    timing_metrics['ranking_avg_time_s'] = ranking_engine.ranking_time / max(ranking_engine.ranking_count, 1)
    timing_metrics['ranking_total_time_s'] = ranking_engine.ranking_time
    codecarbon_metrics['timing'] = timing_metrics

    print("\nCodeCarbon metrics:")
    print(json.dumps(codecarbon_metrics, indent=2))

    all_metrics = {
        'bm25': bm25_metrics,
        'two_stage_reranking': rerank_metrics,
    }

    results_file = save_results(args, all_metrics, codecarbon_metrics, args.output_dir)

    print("\n" + "="*20)
    print("EVALUATION COMPLETE")
    print("="*20)
    print(f"Dataset: {args.dataset}")
    print(f"Filter method: {args.filter_method}")
    print(f"Ranking method: {args.ranking_method}")
    print(f"Results saved to: {results_file}")
    print("="*20)


if __name__ == '__main__':
    main()
