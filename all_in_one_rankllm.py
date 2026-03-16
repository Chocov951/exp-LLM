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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import time
import argparse
import inspect
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModel
from codecarbon import OfflineEmissionsTracker
from beir.datasets.data_loader import GenericDataLoader
from pyserini.search.lucene import LuceneSearcher
from ranx import Qrels, Run, evaluate
import csv
from utils import create_model_config, load_dataset, _convert_to_rankllm_request, _convert_from_rankllm_result, cleanup_stage_engine, fit_filter_passages_to_context, fit_ranking_passages_to_context, _truncate_passages_for_rankllm

# Try to import rank_llm components - will handle gracefully if not installed
RANK_LLM_AVAILABLE = False
CONTEXT_SIZE = 16384
NUM_GPUS = 4
try:
    from rank_llm.rerank import Reranker
    from rank_llm.rerank.listwise import (
        VicunaReranker,
        ZephyrReranker,
    )
    from rank_llm.rerank.pointwise.monot5 import MonoT5
    RANK_LLM_AVAILABLE = True
    try:
        from rank_llm.rerank.listwise import RankListwiseOSLLM
    except ImportError:
        print("Warning: RankListwiseOSLLM not available, will not support custom LLM listwise reranking")


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


# Monkey patch to add top-k filtering to RankListwiseOSLLM
try:
    from monkeypatch import FilterListwiseOSLLM, apply_run_llm_monkeypatch, apply_run_llm_batched_monkeypatch
    apply_run_llm_monkeypatch()
    apply_run_llm_batched_monkeypatch()
except Exception as e:
    print(f"Warning: Could not monkey patch ListwiseRankLLM: {e}")


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
    
    # BM25 index configuration
    parser.add_argument("--bm25_index_path", type=str, default=None,
                       help="Path to a local pyserini Lucene index. "
                            "If not provided, tries pyserini prebuilt BEIR index "
                            "(beir-v1.0.0-{dataset}.flat).")
    
    return parser.parse_args()


# ========================================
# Filter and Ranking Stages
# ========================================


class FilterStage:
    """Filter stage implementation separated from ranking."""

    def __init__(self, args, prompts: Dict[str, str]):
        self.args = args
        self.prompts = prompts
        self.filter_method = args.filter_method
        self.filter_topk = args.filter_topk
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = None
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.vllm_llm = None
        self.vllm_handler = None
        self.vllm_generate_target = None
        self.vllm_seed = int(getattr(args, "vllm_seed", 0))
        self.bert_tokenizer = None
        self.bert_model = None
        self.rankllm_filter = None

        self.filter_time = 0
        self.filter_count = 0

        self._init_models()

    def _init_models(self):
        """Initialize filter models based on configuration."""
        if self.filter_method == "custom_llm":
            self._init_custom_llm()
        elif self.filter_method == "custom_llm_vllm":
            self._init_custom_llm_vllm()
        elif self.filter_method == "bert_index":
            self._init_bert_model()
        elif self.filter_method == "pointwise":
            if RANK_LLM_AVAILABLE:
                self._init_rankllm_pointwise()
            else:
                print("Warning: Pointwise requires rank_llm library")
        elif self.filter_method == "RankLLM_custom":
            if RANK_LLM_AVAILABLE:
                self._init_rankllm_custom()
            else:
                print("Warning: RankLLM_custom filter method requires rank_llm library")

    def _init_custom_llm(self):
        """Initialize custom LLM model for filtering."""
        print(f"Loading custom LLM for filter stage: {self.args.custom_model}")
        model_path, max_length, quantization_config = create_model_config(self.args.custom_model, CONTEXT_SIZE)

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
        self.vllm_llm = None
        self.vllm_handler = None
        self.vllm_generate_target = None
        print("Custom LLM (filter) loaded successfully")

    def _init_custom_llm_vllm(self):
        """Initialize custom LLM model for filtering using rank_llm vLLM handler."""
        print(f"Loading custom LLM vLLM backend for filter stage: {self.args.custom_model}")
        model_path, max_length, quantization_config = create_model_config(self.args.custom_model, CONTEXT_SIZE)
        self.max_length = max_length

        try:
            from vllm import LLM
            self.vllm_llm = LLM(
                model=model_path,
                tensor_parallel_size=NUM_GPUS,
                gpu_memory_utilization=0.75,
                trust_remote_code=True,
                max_model_len=CONTEXT_SIZE,
                seed=self.vllm_seed,
                enforce_eager=False,
                disable_log_stats=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = None
            self.pipe = None
            self.vllm_handler = None
            self.vllm_generate_target = self.vllm_llm
            print("Custom native vLLM backend (filter) loaded successfully")
        except Exception as e:
            print(f"Warning: direct vLLM init failed, trying RankLLM VllmHandler fallback: {e}")
            self.vllm_llm = None

            if not RANK_LLM_AVAILABLE:
                print("Warning: rank_llm not available, falling back to HF custom_llm")
                self._init_custom_llm()
                return

            try:
                from rank_llm.rerank.vllm_handler import VllmHandler

                self.vllm_handler = VllmHandler(
                    model=model_path,
                    download_dir=os.getenv("HF_HOME"),
                    enforce_eager=False,
                    max_logprobs=30,
                    tensor_parallel_size=NUM_GPUS,
                    gpu_memory_utilization=0.75,
                    trust_remote_code=True,
                    max_model_len=CONTEXT_SIZE,
                    disable_log_stats=True,
                )
                self.tokenizer = self.vllm_handler.get_tokenizer()
                self.model = None
                self.pipe = None
                self.vllm_generate_target = self._discover_vllm_generate_target(self.vllm_handler)
                print("Custom RankLLM vLLM backend (filter) loaded successfully")
                if self.vllm_generate_target is None:
                    print("Warning: no direct generate target discovered on VllmHandler at init")
            except Exception as e2:
                print(f"Error initializing custom LLM vLLM backend: {e2}")
                print("Falling back to HF custom LLM backend")
                self._init_custom_llm()

    def _discover_vllm_generate_target(self, handler: Any) -> Optional[Any]:
        """Find an object exposing a vLLM-like `.generate(...)` method."""
        if handler is None:
            return None

        if hasattr(handler, "generate") and callable(getattr(handler, "generate")):
            return handler

        candidate_attrs = (
            "_llm", "llm", "_engine", "engine", "_model", "model", "_client", "client"
        )
        for attr_name in candidate_attrs:
            candidate = getattr(handler, attr_name, None)
            if candidate is not None and hasattr(candidate, "generate") and callable(getattr(candidate, "generate")):
                return candidate

        return None

    def _extract_generated_text(self, raw_output: Any) -> str:
        """Extract generated text from varying HF/vLLM response shapes."""
        if raw_output is None:
            return ""

        if isinstance(raw_output, str):
            return raw_output

        if isinstance(raw_output, list):
            if not raw_output:
                return ""

            first = raw_output[0]
            if isinstance(first, dict):
                if "generated_text" in first:
                    generated = first["generated_text"]
                    if isinstance(generated, list) and generated and isinstance(generated[-1], dict):
                        return generated[-1].get("content", "")
                    if isinstance(generated, str):
                        return generated
                if "text" in first:
                    return str(first["text"])

            if hasattr(first, "outputs") and first.outputs:
                out0 = first.outputs[0]
                if hasattr(out0, "text"):
                    return str(out0.text)

            return str(first)

        if isinstance(raw_output, dict):
            if "generated_text" in raw_output:
                return str(raw_output["generated_text"])
            if "text" in raw_output:
                return str(raw_output["text"])

        if hasattr(raw_output, "outputs") and raw_output.outputs:
            out0 = raw_output.outputs[0]
            if hasattr(out0, "text"):
                return str(out0.text)

        return str(raw_output)

    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build plain prompt text from chat messages for backends without native chat mode."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        system_prompt = messages[0].get("content", "") if messages else ""
        user_prompt = messages[1].get("content", "") if len(messages) > 1 else ""
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    def _generate_with_custom_backend(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        """Generate text with HF pipeline or vLLM handler while tolerating API differences."""
        def _build_vllm_sampling_params(max_tokens: int):
            from vllm import SamplingParams
            return SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                seed=self.vllm_seed,
            )

        if self.pipe is not None:
            outputs = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
            )
            return self._extract_generated_text(outputs)

        if self.vllm_handler is None:
            if self.vllm_llm is None:
                raise RuntimeError("No custom LLM backend initialized")

        prompt = self._build_chat_prompt(messages)

        if self.vllm_llm is not None:
            sampling_params = _build_vllm_sampling_params(max_new_tokens)
            raw_output = self.vllm_llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            text = self._extract_generated_text(raw_output)
            if text:
                return text

            raise RuntimeError("Native vLLM generation returned empty text")

        self.vllm_generate_target = self.vllm_generate_target or self._discover_vllm_generate_target(self.vllm_handler)
        attempts = []

        underlying_llm = self.vllm_generate_target or getattr(self.vllm_handler, "_llm", None)
        if underlying_llm is not None and hasattr(underlying_llm, "generate"):
            attempts.append(("_llm.generate", ([prompt],), {}))

        if hasattr(self.vllm_handler, "chat"):
            attempts.append(("chat", (), {"messages": messages, "max_new_tokens": max_new_tokens}))
            attempts.append(("chat", (messages,), {"max_new_tokens": max_new_tokens}))

        if hasattr(self.vllm_handler, "run_llm"):
            attempts.append(("run_llm", (prompt,), {"max_new_tokens": max_new_tokens}))
            attempts.append(("run_llm", (prompt,), {}))

        if hasattr(self.vllm_handler, "generate"):
            attempts.append(("generate", (prompt,), {"max_new_tokens": max_new_tokens}))
            attempts.append(("generate", ([prompt],), {"max_new_tokens": max_new_tokens}))

        if callable(self.vllm_handler):
            attempts.append(("__call__", (prompt,), {"max_new_tokens": max_new_tokens}))

        last_error = None
        tried_methods = []
        for method_name, method_args, method_kwargs in attempts:
            tried_methods.append(method_name)
            try:
                if method_name == "_llm.generate":
                    sampling_params = _build_vllm_sampling_params(max_new_tokens)
                    raw_output = underlying_llm.generate(
                        method_args[0],
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                else:
                    method = self.vllm_handler if method_name == "__call__" else getattr(self.vllm_handler, method_name)
                    if method_kwargs:
                        try:
                            sig = inspect.signature(method)
                            accepted_kwargs = {
                                key: value for key, value in method_kwargs.items() if key in sig.parameters
                            }
                        except Exception:
                            accepted_kwargs = method_kwargs
                    else:
                        accepted_kwargs = method_kwargs

                    raw_output = method(*method_args, **accepted_kwargs)

                text = self._extract_generated_text(raw_output)
                if text:
                    return text
            except Exception as e:
                last_error = e

        if not attempts:
            raise RuntimeError(
                "No compatible generation method found on vLLM handler. "
                f"Handler type: {type(self.vllm_handler)}; attrs={list(vars(self.vllm_handler).keys()) if hasattr(self.vllm_handler, '__dict__') else 'n/a'}"
            )

        raise RuntimeError(
            "All vLLM generation attempts failed or returned empty text. "
            f"Tried methods: {tried_methods}. Last error: {last_error}"
        )

    def _fit_filter_passages_to_context(self, query: str, passages: Dict[str, str]) -> Dict[str, str]:
        """Shrink filter passages so prompt fits model context window."""
        return fit_filter_passages_to_context(
            query=query,
            passages=passages,
            prompts=self.prompts,
            tokenizer=self.tokenizer,
            context_size=CONTEXT_SIZE,
            filter_topk=self.filter_topk,
        )

    def _init_bert_model(self):
        """Initialize BERT model for filtering."""
        print(f"Loading BERT model: {self.args.bert_model}")
        model_path = os.path.join("models", self.args.bert_model)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert_model = AutoModel.from_pretrained(model_path).to(self.device)
        print("BERT model loaded successfully")

    def _ensure_bert_loaded(self):
        if self.bert_tokenizer is None or self.bert_model is None:
            self._init_bert_model()

    def _init_rankllm_pointwise(self):
        """Initialize RankLLM pointwise reranker."""
        if not RANK_LLM_AVAILABLE:
            print("Warning: rank_llm not available, cannot initialize pointwise reranker")
            return
        
        print("Loading RankLLM pointwise reranker")
        try:
            # PointwiseRankLLM for pointwise ranking
            # Local model directory
            llm_root = os.environ.get("LLM_MODELS_ROOT")
            self.rankllm_filter = Reranker(MonoT5(
                model=os.path.join(llm_root, "monot5-base-msmarco-10k") if llm_root else "models/monot5-base-msmarco-10k",
                context_size=CONTEXT_SIZE,
                device=self.device,
            ))
            print("RankLLM pointwise reranker loaded successfully")
        except Exception as e:
            print(f"Error initializing RankLLM pointwise reranker: {e}")
            print("Falling back to custom LLM")
            self.rankllm_filter = None

    def _init_rankllm_custom(self):
        """Initialize RankLLM custom filter."""
        if not RANK_LLM_AVAILABLE:
            print("Warning: rank_llm not available, cannot initialize custom filter")
            return

        print("Loading RankLLM custom filter")
        window_size = self.args.bm25_topk
        try:
            model_path, max_length, quantization_config = create_model_config(self.args.custom_model, CONTEXT_SIZE)
            self.rankllm_filter = FilterListwiseOSLLM(
                model=model_path,
                window_size=window_size,
                stride=self.args.listwise_stride,
                n_keep=self.filter_topk,
                strict_exact_n=True,
                context_size=CONTEXT_SIZE,
                num_gpus=NUM_GPUS,
                device=self.device,
                gpu_memory_utilization=0.75,
            )
            print("RankLLM custom filter loaded successfully")
        except Exception as e:
            print(f"Error initializing RankLLM custom filter: {e}")
            print("Falling back to custom LLM")
            self.rankllm_filter = None

    def filter(self, query: str, passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """Run the filter stage and return a relevance score dict."""
        start_time = time.time()

        if self.filter_method in ("custom_llm", "custom_llm_vllm"):
            result = self._filter_custom_llm(query, passages)
        elif self.filter_method == "bert_index":
            result = self._filter_bert(query, passages)
        elif self.filter_method == "pointwise":
            result = self._filter_pointwise(query, passages, qid)
        elif self.filter_method == "RankLLM_custom":
            result = self._filter_rankllm_custom(query, passages, qid)
        else:
            raise ValueError(f"Unknown filter method: {self.filter_method}")

        self.filter_time += time.time() - start_time
        self.filter_count += 1
        return result

    def _filter_custom_llm(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Filter using custom LLM with prompts."""
        passages = self._fit_filter_passages_to_context(query, passages)
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
            llm_rep = self._generate_with_custom_backend(
                messages,
                max_new_tokens=10*self.filter_topk+10,
            )
            print(f"\n\nRaw LLM output for filtering:\n{llm_rep}")

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
        self._ensure_bert_loaded()
        query_inputs = self.bert_tokenizer(query, return_tensors="pt",
                                          truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            query_outputs = self.bert_model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state[:, 0, :].cpu()

        similarities = {}
        for doc_id, passage_text in passages.items():
            passage_inputs = self.bert_tokenizer(passage_text, return_tensors="pt",
                                                truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                passage_outputs = self.bert_model(**passage_inputs)
                passage_embedding = passage_outputs.last_hidden_state[:, 0, :].cpu()

            similarity = torch.nn.functional.cosine_similarity(
                query_embedding, passage_embedding
            ).item()
            similarities[doc_id] = similarity

        return similarities

    def _filter_pointwise(self, query: str, passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """Filter using Pointwise from rank_llm."""
        if not RANK_LLM_AVAILABLE or self.rankllm_filter is None:
            print("Pointwise filtering not available, falling back to BERT")
            return self._filter_bert(query, passages)

        try:
            # Convert to RankLLM format
            request = _convert_to_rankllm_request(query, passages, qid)
            
            # Rerank using RankLLM (handle API differences)
            result = None
            if hasattr(self.rankllm_filter, "rerank"):
                result = self.rankllm_filter.rerank(
                    request=request,
                    rank_start=0,
                    rank_end=len(passages)
                )
            elif hasattr(self.rankllm_filter, "rank"):
                result = self.rankllm_filter.rank(
                    request=request,
                    rank_start=0,
                    rank_end=len(passages)
                )
            elif callable(self.rankllm_filter):
                result = self.rankllm_filter(request)

            if result is None:
                raise AttributeError("RankLLM pointwise reranker exposes no ranking method (rerank/rank/__call__)")
            
            # Convert back to our format
            return _convert_from_rankllm_result(result)
        
        except Exception as e:
            print(f"Error in RankLLM pointwise filtering: {e}")
            print("Falling back to BERT filtering")
            return self._filter_bert(query, passages)
        
    def _filter_rankllm_custom(self, query: str, passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """Filter using custom filter from rank_llm."""
        if not RANK_LLM_AVAILABLE or self.rankllm_filter is None:
            print("Custom filtering not available, falling back to BERT")
            return self._filter_bert(query, passages)

        try:
            # Convert to RankLLM format
            request = _convert_to_rankllm_request(query, passages, qid)
            
            # Rerank using RankLLM (handle API differences)
            result = None
            
            result = self.rankllm_filter.rerank_batch(
                requests=[request],
                rank_start=0,
                rank_end=len(passages)
            )

            if result is None:
                raise AttributeError("RankLLM custom filter exposes no ranking method (rerank/rank/__call__)")
            
            # Convert back to our format
            return _convert_from_rankllm_result(result)
        
        except Exception as e:
            print(f"Error in RankLLM custom filtering: {e}")
            print("Falling back to BERT filtering")
            return self._filter_bert(query, passages)

    def export_shared_llm(self) -> Optional[Dict[str, object]]:
        """Expose custom LLM components for reuse by ranking stage."""
        if self.filter_method not in ("custom_llm", "custom_llm_vllm"):
            return None
        return {
            "pipe": self.pipe,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_length": self.max_length,
            "vllm_llm": self.vllm_llm,
            "vllm_handler": self.vllm_handler,
            "vllm_generate_target": self.vllm_generate_target,
        }


class RankingStage:
    """Ranking stage implementation separated from filtering."""

    def __init__(self, args, prompts: Dict[str, str], shared_custom_llm: Optional[Dict[str, object]] = None):
        self.args = args
        self.prompts = prompts
        self.ranking_method = args.ranking_method
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_length = None
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.vllm_llm = None
        self.vllm_handler = None
        self.vllm_generate_target = None
        self.vllm_seed = int(getattr(args, "vllm_seed", 0))
        
        # RankLLM rerankers
        self.rankllm_reranker = None

        self.ranking_time = 0
        self.ranking_count = 0

        self._init_models(shared_custom_llm)

    def _init_models(self, shared_custom_llm: Optional[Dict[str, object]]):
        """Initialize ranking models based on configuration."""
        if self.ranking_method in ("custom_llm", "custom_llm_vllm"):
            if shared_custom_llm and (
                shared_custom_llm.get("pipe")
                or shared_custom_llm.get("vllm_handler")
                or shared_custom_llm.get("vllm_llm")
            ):
                print("Reusing custom LLM from filter stage for ranking")
                self.pipe = shared_custom_llm.get("pipe")
                self.model = shared_custom_llm.get("model")
                self.tokenizer = shared_custom_llm.get("tokenizer")
                self.max_length = shared_custom_llm.get("max_length")
                self.vllm_llm = shared_custom_llm.get("vllm_llm")
                self.vllm_handler = shared_custom_llm.get("vllm_handler")
                self.vllm_generate_target = shared_custom_llm.get("vllm_generate_target")
            else:
                if self.ranking_method == "custom_llm_vllm":
                    self._init_custom_llm_vllm()
                else:
                    self._init_custom_llm()
        elif self.ranking_method == "listwise":
            if self.args.listwise_model == "custom_llm":
                model_path, max_length, quantization_config = create_model_config(self.args.custom_model, CONTEXT_SIZE)
            else:
                model_path = None
            self._init_rankllm_listwise(model=model_path)

    def _init_custom_llm(self):
        """Initialize custom LLM model for ranking."""
        print(f"Loading custom LLM for ranking stage: {self.args.custom_model}")
        model_path, max_length, quantization_config = create_model_config(self.args.custom_model, CONTEXT_SIZE)

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
        self.vllm_llm = None
        self.vllm_handler = None
        self.vllm_generate_target = None
        print("Custom LLM (ranking) loaded successfully")

    def _init_custom_llm_vllm(self):
        """Initialize custom LLM model for ranking using rank_llm vLLM handler."""
        print(f"Loading custom LLM vLLM backend for ranking stage: {self.args.custom_model}")
        model_path, max_length, quantization_config = create_model_config(self.args.custom_model, CONTEXT_SIZE)
        self.max_length = max_length

        try:
            from vllm import LLM
            self.vllm_llm = LLM(
                model=model_path,
                tensor_parallel_size=NUM_GPUS,
                gpu_memory_utilization=0.75,
                trust_remote_code=True,
                max_model_len=CONTEXT_SIZE,
                seed=self.vllm_seed,
                enforce_eager=False,
                disable_log_stats=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = None
            self.pipe = None
            self.vllm_handler = None
            self.vllm_generate_target = self.vllm_llm
            print("Custom native vLLM backend (ranking) loaded successfully")
        except Exception as e:
            print(f"Warning: direct vLLM init failed, trying RankLLM VllmHandler fallback: {e}")
            self.vllm_llm = None

            if not RANK_LLM_AVAILABLE:
                print("Warning: rank_llm not available, falling back to HF custom_llm")
                self._init_custom_llm()
                return

            try:
                from rank_llm.rerank.vllm_handler import VllmHandler

                self.vllm_handler = VllmHandler(
                    model=model_path,
                    download_dir=os.getenv("HF_HOME"),
                    enforce_eager=False,
                    max_logprobs=30,
                    tensor_parallel_size=NUM_GPUS,
                    gpu_memory_utilization=0.75,
                    trust_remote_code=True,
                    max_model_len=CONTEXT_SIZE,
                    disable_log_stats=True,
                )
                self.tokenizer = self.vllm_handler.get_tokenizer()
                self.model = None
                self.pipe = None
                self.vllm_generate_target = self._discover_vllm_generate_target(self.vllm_handler)
                print("Custom RankLLM vLLM backend (ranking) loaded successfully")
                if self.vllm_generate_target is None:
                    print("Warning: no direct generate target discovered on VllmHandler at init")
            except Exception as e2:
                print(f"Error initializing custom LLM vLLM backend: {e2}")
                print("Falling back to HF custom LLM backend")
                self._init_custom_llm()

    def _discover_vllm_generate_target(self, handler: Any) -> Optional[Any]:
        """Find an object exposing a vLLM-like `.generate(...)` method."""
        if handler is None:
            return None

        if hasattr(handler, "generate") and callable(getattr(handler, "generate")):
            return handler

        candidate_attrs = (
            "_llm", "llm", "_engine", "engine", "_model", "model", "_client", "client"
        )
        for attr_name in candidate_attrs:
            candidate = getattr(handler, attr_name, None)
            if candidate is not None and hasattr(candidate, "generate") and callable(getattr(candidate, "generate")):
                return candidate

        return None

    def _extract_generated_text(self, raw_output: Any) -> str:
        """Extract generated text from varying HF/vLLM response shapes."""
        if raw_output is None:
            return ""

        if isinstance(raw_output, str):
            return raw_output

        if isinstance(raw_output, list):
            if not raw_output:
                return ""

            first = raw_output[0]
            if isinstance(first, dict):
                if "generated_text" in first:
                    generated = first["generated_text"]
                    if isinstance(generated, list) and generated and isinstance(generated[-1], dict):
                        return generated[-1].get("content", "")
                    if isinstance(generated, str):
                        return generated
                if "text" in first:
                    return str(first["text"])

            if hasattr(first, "outputs") and first.outputs:
                out0 = first.outputs[0]
                if hasattr(out0, "text"):
                    return str(out0.text)

            return str(first)

        if isinstance(raw_output, dict):
            if "generated_text" in raw_output:
                return str(raw_output["generated_text"])
            if "text" in raw_output:
                return str(raw_output["text"])

        if hasattr(raw_output, "outputs") and raw_output.outputs:
            out0 = raw_output.outputs[0]
            if hasattr(out0, "text"):
                return str(out0.text)

        return str(raw_output)

    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build plain prompt text from chat messages for backends without native chat mode."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        system_prompt = messages[0].get("content", "") if messages else ""
        user_prompt = messages[1].get("content", "") if len(messages) > 1 else ""
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    def _generate_with_custom_backend(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        """Generate text with HF pipeline or vLLM handler while tolerating API differences."""
        def _build_vllm_sampling_params(max_tokens: int):
            from vllm import SamplingParams
            return SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                seed=self.vllm_seed,
            )

        if self.pipe is not None:
            outputs = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
            )
            return self._extract_generated_text(outputs)

        if self.vllm_handler is None:
            if self.vllm_llm is None:
                raise RuntimeError("No custom LLM backend initialized")

        prompt = self._build_chat_prompt(messages)

        if self.vllm_llm is not None:
            sampling_params = _build_vllm_sampling_params(max_new_tokens)
            raw_output = self.vllm_llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            text = self._extract_generated_text(raw_output)
            if text:
                return text

            raise RuntimeError("Native vLLM generation returned empty text")

        self.vllm_generate_target = self.vllm_generate_target or self._discover_vllm_generate_target(self.vllm_handler)
        attempts = []

        underlying_llm = self.vllm_generate_target or getattr(self.vllm_handler, "_llm", None)
        if underlying_llm is not None and hasattr(underlying_llm, "generate"):
            attempts.append(("_llm.generate", ([prompt],), {}))

        if hasattr(self.vllm_handler, "chat"):
            attempts.append(("chat", (), {"messages": messages, "max_new_tokens": max_new_tokens}))
            attempts.append(("chat", (messages,), {"max_new_tokens": max_new_tokens}))

        if hasattr(self.vllm_handler, "run_llm"):
            attempts.append(("run_llm", (prompt,), {"max_new_tokens": max_new_tokens}))
            attempts.append(("run_llm", (prompt,), {}))

        if hasattr(self.vllm_handler, "generate"):
            attempts.append(("generate", (prompt,), {"max_new_tokens": max_new_tokens}))
            attempts.append(("generate", ([prompt],), {"max_new_tokens": max_new_tokens}))

        if callable(self.vllm_handler):
            attempts.append(("__call__", (prompt,), {"max_new_tokens": max_new_tokens}))

        last_error = None
        tried_methods = []
        for method_name, method_args, method_kwargs in attempts:
            tried_methods.append(method_name)
            try:
                if method_name == "_llm.generate":
                    sampling_params = _build_vllm_sampling_params(max_new_tokens)
                    raw_output = underlying_llm.generate(
                        method_args[0],
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                else:
                    method = self.vllm_handler if method_name == "__call__" else getattr(self.vllm_handler, method_name)
                    if method_kwargs:
                        try:
                            sig = inspect.signature(method)
                            accepted_kwargs = {
                                key: value for key, value in method_kwargs.items() if key in sig.parameters
                            }
                        except Exception:
                            accepted_kwargs = method_kwargs
                    else:
                        accepted_kwargs = method_kwargs

                    raw_output = method(*method_args, **accepted_kwargs)

                text = self._extract_generated_text(raw_output)
                if text:
                    return text
            except Exception as e:
                last_error = e

        if not attempts:
            raise RuntimeError(
                "No compatible generation method found on vLLM handler. "
                f"Handler type: {type(self.vllm_handler)}; attrs={list(vars(self.vllm_handler).keys()) if hasattr(self.vllm_handler, '__dict__') else 'n/a'}"
            )

        raise RuntimeError(
            "All vLLM generation attempts failed or returned empty text. "
            f"Tried methods: {tried_methods}. Last error: {last_error}"
        )

    def _fit_ranking_passages_to_context(self, query: str, passages: Dict[str, str]) -> Dict[str, str]:
        """Shrink ranking passages so prompt fits model context window."""
        return fit_ranking_passages_to_context(
            query=query,
            passages=passages,
            prompts=self.prompts,
            tokenizer=self.tokenizer,
            context_size=CONTEXT_SIZE,
        )
    
    def _init_rankllm_listwise(self, model: Optional[str] = None):
        """Initialize RankLLM listwise reranker."""
        if not RANK_LLM_AVAILABLE:
            print("Warning: rank_llm not available, cannot initialize listwise reranker")
            return
        
        print(f"Loading RankLLM listwise reranker: {self.args.listwise_model}")
        try:
            if self.args.listwise_window_size == "full":
                print("Using RankLLM full window size for single-pass reranking")
                window_size = self.args.filter_topk
            else:
                window_size = int(self.args.listwise_window_size)
            
            # Local model directory
            llm_root = os.environ.get("LLM_MODELS_ROOT")
            if "zephyr" in self.args.listwise_model.lower():
                self.rankllm_reranker = ZephyrReranker(
                    model_path=os.path.join(llm_root, "rank_zephyr_7b_v1_full") if llm_root else "models/rank_zephyr_7b_v1_full",
                    context_size=CONTEXT_SIZE,
                    num_gpus=NUM_GPUS,
                    device=self.device,
                    window_size=window_size,
                    stride=self.args.listwise_stride,
                )
            elif "vicuna" in self.args.listwise_model.lower():
                self.rankllm_reranker = VicunaReranker(
                    model_path=os.path.join(llm_root, "rank_vicuna_7b_v1") if llm_root else "models/rank_vicuna_7b_v1",
                    context_size=CONTEXT_SIZE,
                    num_gpus=NUM_GPUS,
                    device=self.device,
                    window_size=window_size,
                    stride=self.args.listwise_stride,
                )
            else:
                # Use custom tranformers model with RankListwiseOSLLM
                self.rankllm_reranker = RankListwiseOSLLM(
                    model=model,
                    context_size=CONTEXT_SIZE,
                    num_gpus=NUM_GPUS,
                    device=self.device,
                    window_size=window_size,
                    stride=self.args.listwise_stride,
                )
                # from monkeypatch import RankListwiseOSLLM_Capped
                # self.rankllm_reranker = RankListwiseOSLLM_Capped(
                #     model=model,
                #     context_size=CONTEXT_SIZE,
                #     num_gpus=NUM_GPUS,
                #     device=self.device,
                #     window_size=window_size,
                #     stride=self.args.listwise_stride,
                #     gpu_memory_utilization=0.75,
                # )
            print("RankLLM listwise reranker loaded successfully")

            if self.args.listwise_model.lower() in ("zephyr", "vicuna"):
                reranker_obj = getattr(self.rankllm_reranker, "_reranker", None)
                tokenizer_obj = getattr(reranker_obj, "_tokenizer", None) if reranker_obj is not None else None
                effective_context_size = getattr(reranker_obj, "_context_size", None) if reranker_obj is not None else None
                tokenizer_max_len = getattr(tokenizer_obj, "model_max_length", None) if tokenizer_obj is not None else None

                print(
                    "[startup] effective listwise context | "
                    f"model={self.args.listwise_model} | "
                    f"requested_context_size={CONTEXT_SIZE} | "
                    f"reranker_context_size={effective_context_size} | "
                    f"tokenizer_model_max_length={tokenizer_max_len}"
                )
        except Exception as e:
            print(f"Error initializing RankLLM listwise reranker: {e}")
            print("Falling back to custom LLM")
            self.rankllm_reranker = None

    def rank(self, query: str, filtered_passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """Run the ranking stage and return rank scores."""
        start_time = time.time()

        if self.ranking_method in ("custom_llm", "custom_llm_vllm"):
            result = self._rank_custom_llm(query, filtered_passages)
        elif self.ranking_method == "listwise":
            result = self._rank_rankllm_listwise(query, filtered_passages, qid)
        else:
            raise ValueError(f"Unknown ranking method: {self.ranking_method}")

        self.ranking_time += time.time() - start_time
        self.ranking_count += 1
        return result

    def _rank_custom_llm(self, query: str, passages: Dict[str, str]) -> Dict[str, float]:
        """Rank using custom LLM with prompts."""
        passages = self._fit_ranking_passages_to_context(query, passages)
        messages = [
            {"role": "system", "content": self.prompts['rerank-S-prompt']},
            {"role": "user", "content":
                f"{self.prompts['rerank-U-prompt-1']}{json.dumps(passages)}\n\n"
                f"{self.prompts['rerank-U-prompt-2']}{query}\n\n"
                f"{self.prompts['rerank-U-prompt-3']}"
            }
        ]

        try:
            llm_rep = self._generate_with_custom_backend(
                messages,
                max_new_tokens=10*len(passages)+10,
            )

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

    def _rank_rankllm_listwise(self, query: str, passages: Dict[str, str], qid: str) -> Dict[str, float]:
        """Rank using rank_llm listwise ranker."""
        if not RANK_LLM_AVAILABLE or self.rankllm_reranker is None:
            print("Warning: rank_llm not available, falling back to custom LLM")
            return self._rank_custom_llm(query, passages)

        try:
            if self.args.listwise_model != "custom_llm":
                passages = _truncate_passages_for_rankllm(query, passages, self.rankllm_reranker)

            # Convert to RankLLM format
            request = _convert_to_rankllm_request(query, passages, qid)
            
            # Rerank using RankLLM while tolerating API differences between versions
            result = None
            if self.args.listwise_model != "custom_llm":
                result = self.rankllm_reranker.rerank(
                    request=request,
                    rank_start=0,
                    rank_end=len(passages)
                )
            else:
                result = self.rankllm_reranker.rerank_batch(
                    requests=[request],
                    rank_start=0,
                    rank_end=len(passages)
                )
            # print(f"Parsed RankLLM output: {result}")

            if result is None:
                raise AttributeError("RankLLM reranker exposes no ranking method (rerank/rank/__call__)")

            # Convert back to our format
            return _convert_from_rankllm_result(result)
        except Exception as e:
            print(f"Error in RankLLM listwise ranking: {e}")
            print("Falling back to custom LLM")
            return self._rank_custom_llm(query, passages)
        

# ========================================
# Main Pipeline Functions
# ========================================


def load_bm25_results(dataset: str, bm25_topk: int,
                      queries: Optional[Dict] = None,
                      corpus: Optional[Dict] = None,
                      index_path: Optional[str] = None) -> Tuple[Dict, Dict]:
    """
    Load or compute BM25 results using pyserini.

    If pre-cached JSON files exist they are returned immediately.
    Otherwise, pyserini is used to search the index (either the path given by
    *index_path* or the pyserini prebuilt BEIR index for *dataset*).

    Args:
        dataset: dataset name (e.g. "scifact", "fiqa")
        bm25_topk: number of documents to retrieve per query
        queries: {qid: query_text} mapping (needed when cache is missing)
        corpus: {doc_id: text} mapping (needed to populate passage texts)
        index_path: path to a local pyserini Lucene index; if None, the
                    prebuilt BEIR index ``beir-v1.0.0-{dataset}.flat`` is used.

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
    
    print(f"Computing BM25 results for {dataset} using pyserini...")

    if not queries:
        print("Error: queries must be provided to compute BM25 results.")
        return {}, {}

    try:
        if index_path:
            print(f"Opening local pyserini index: {index_path}")
            searcher = LuceneSearcher(index_path)
        else:
            prebuilt_name = f'beir-v1.0.0-{dataset}.flat'
            print(f"Loading pyserini prebuilt index: {prebuilt_name}")
            searcher = LuceneSearcher.from_prebuilt_index(prebuilt_name)
    except Exception as e:
        print(f"Error opening pyserini index: {e}")
        print("Provide --bm25_index_path or ensure the dataset has a prebuilt pyserini index.")
        return {}, {}

    bm25_rundict = {}
    queries_with_passages = {}

    for qid, query_text in tqdm(queries.items(), desc="BM25 search"):
        try:
            hits = searcher.search(query_text, k=bm25_topk)
        except Exception as e:
            print(f"Error searching query {qid}: {e}")
            continue

        doc_scores = {}
        passages = {}
        for hit in hits:
            docid = str(hit.docid)
            doc_scores[docid] = float(hit.score)

            # Retrieve text: prefer local corpus, fall back to index raw content
            if corpus and docid in corpus:
                text = corpus[docid]
            elif corpus and docid.lstrip('d') in corpus:
                text = corpus[docid.lstrip('d')]
            else:
                try:
                    raw = json.loads(searcher.doc(hit.docid).raw())
                    text = raw.get('contents', raw.get('text', ''))
                except Exception:
                    text = ''
            passages[docid] = text

        bm25_rundict[str(qid)] = doc_scores
        queries_with_passages[str(qid)] = {'query': query_text, 'passages': passages}

    # Persist to cache
    os.makedirs('rundicts', exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(bm25_rundict, f, indent=2)
    with open(queries_file, 'w', encoding='utf-8') as f:
        json.dump(queries_with_passages, f, indent=2)
    print(f"BM25 results saved to {cache_file}")

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
        print(f"Dataset not found locally ({e}). Trying BEIR auto-download...")
        try:
            from beir import util as beir_util
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
            beir_util.download_and_unzip(url, "datasets")
            corpus, queries, qrels = GenericDataLoader(f"datasets/{args.dataset}").load(split="test")
            corpus = {k: v['text'] if isinstance(v, dict) else v for k, v in corpus.items()}
            print(f"Downloaded and loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels")
        except Exception as e2:
            print(f"Error loading dataset: {e2}")
            print("Please ensure the dataset is available in datasets/ directory or can be downloaded from BEIR.")
            return
    
    # Load BM25 results (needed for filtering stage)
    bm25_rundict, queries_with_passages = load_bm25_results(
        args.dataset, args.bm25_topk,
        queries=queries, corpus=corpus,
        index_path=args.bm25_index_path
    )

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
            query_text = queries_with_passages[qid]['query']
            corpus = queries_with_passages[qid]['passages']
            corpus_texts = list(corpus.values())
            corpus_ids = list(corpus.keys())

            if str(qid) not in bm25_rundict and qid not in bm25_rundict:
                print(f"Warning: Query {qid} not in BM25 results")
                continue

            bm25_qid = str(qid) if str(qid) in bm25_rundict else qid
            top_k_passage_ids = list(bm25_rundict[bm25_qid].keys())[:args.bm25_topk]
           
            passages = {}
            for pid in top_k_passage_ids:
                if pid in corpus_ids:
                    passages[pid] = corpus[pid]
                elif str(pid) in corpus_ids:
                    passages[str(pid)] = corpus[str(pid)]

            if not passages:
                print(f"Warning: No passages found for query {qid}")
                continue

            try:
                filter_scores = filter_engine.filter(query_text, passages, str(qid))
                if not filter_scores:
                    print(f"Warning: No filter scores returned for query {qid}")
                    continue
                sorted_passages = sorted(filter_scores.items(), key=lambda x: x[1], reverse=True)
                filtered_rundict[str(qid)] = {pid: score for pid, score in sorted_passages}
            except Exception as e:
                print(f"Error filtering query {qid}: {e}")
                continue
        filter_avg_time = filter_engine.filter_time / max(filter_engine.filter_count, 1)
        filter_total_time = filter_engine.filter_time

        tracker_filter.stop()
        save_rundict(filtered_rundict, filtered_output_path)

        cleanup_stage_engine(filter_engine, "rankllm_filter")
        filter_engine = None

        import subprocess
        print(subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.free","--format=csv"]).decode())

        # No early return here; filter-only runs will skip ranking later but still report metrics

    if args.stage == "rerank":
        if args.load_filtered_rundict:
            load_path = args.load_filtered_rundict
            if not os.path.exists(load_path):
                print(f"Error: Provided filtered rundict not found at {load_path}")
                return
            with open(load_path, 'r', encoding='utf-8') as f:
                filtered_rundict = json.load(f)
        else:
            print(f"Using top{args.bm25_topk} BM25 results as filtered rundict for reranking")
            filtered_rundict = {}
            for qid in queries_to_process:
                bm25_qid = str(qid) if str(qid) in bm25_rundict else qid
                top_k_passage_ids = list(bm25_rundict[bm25_qid].keys())[:args.bm25_topk]
                filtered_rundict[bm25_qid] = {pid: bm25_rundict[bm25_qid][pid] for pid in top_k_passage_ids}

    rundict_rerank = {}

    if args.stage in ("both", "rerank"):
        if not filtered_rundict:
            print("Error: No filtered rundict available for ranking stage")
            return

        print("\nRunning ranking stage...")
        ranking_engine = RankingStage(args, prompts)
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

        print(f"\n\nOUTPUT START")
        for qid in tqdm(queries_to_process, desc="Reranking queries"):
            query_text = queries_with_passages[qid]['query']
            corpus = queries_with_passages[qid]['passages']
            corpus_texts = list(corpus.values())
            corpus_ids = list(corpus.keys())
            filtered_scores = filtered_rundict.get(str(qid)) or filtered_rundict.get(qid)
            if not filtered_scores:
                print(f"Warning: No filtered docs for query {qid}")
                sorted_filtered = {}
                continue

            else:
                # print(f"size of filtered scores before topk: {len(filtered_scores)}")
                filtered_scores = {pid: score for pid, score in list(filtered_scores.items())[:args.filter_topk]}
                sorted_filtered = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)

            filtered_passages = {}
            for pid, _ in sorted_filtered:
                if pid in corpus_ids:
                    filtered_passages[pid] = corpus[pid]
                elif str(pid) in corpus_ids:
                    filtered_passages[str(pid)] = corpus[str(pid)]

            if not filtered_passages:
                print(f"Warning: No passages found for reranking query {qid}")
                continue

            try:
                # print(f"size of filtered passages after topk: {len(filtered_passages)}")
                final_scores = ranking_engine.rank(query_text, filtered_passages, str(qid))
                rundict_rerank[str(qid)] = final_scores
            except Exception as e:
                print(f"Error reranking query {qid}: {e}")
                continue

        tracker_ranking.stop()
        save_rundict(rundict_rerank, ranking_output_path)
    else:
        ranking_engine = None
        tracker_ranking = None

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
