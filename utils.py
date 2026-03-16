import os
import gzip
import json
import torch
from typing import Tuple, Optional, Dict, List
from beir.datasets.data_loader import GenericDataLoader
from transformers import BitsAndBytesConfig
from rank_llm.data import Request, Query, Candidate, Result

def create_model_config(model_name: str, context_size: int) -> Tuple[str, int, Optional[BitsAndBytesConfig]]:
    """Create model configuration based on model name."""

    model_configs = {
        'qwen4': ('Qwen3-4B-Instruct-2507', context_size, True),
        'qwen30': ('Qwen3-30B-A3B-Instruct-2507', context_size, True),
        'qwen72': ('Qwen2.5-72B-Instruct', context_size, True),
        'smollm': ('SmolLM3-3B', context_size, True),
        'gpt': ('gpt-oss-20b', context_size, True),
        'deepseek': ('DeepSeek-V3.2', context_size, True)
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Model {model_name} not supported")
    
    model_path, max_length, use_quant = model_configs[model_name]
    # Local model directory
    llm_root = os.environ.get("LLM_MODELS_ROOT")
    if llm_root:
        model_path = os.path.join(llm_root, model_path)
    else:
        model_path = os.path.join("models", model_path)
    
    quantization_config = None
    if use_quant:
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            print("Warning: CUDA not available; disabling 8-bit quantization for CPU run.")
            quantization_config = None
    
    return model_path, max_length, quantization_config

def load_dataset(dataset: str) -> Tuple[Dict, Dict, Dict]:
    # Load dataset    
    if dataset == 'trec19' or dataset == 'trec20':
        corpus_test, queries_test, qrels_test = {}, {}, {}
        year = dataset[-2:]
        # with tarfile.open("datasets/trec/collection.tar.gz", "r:gz") as tar:
        #     for member in tar:
        #         if member.isfile() and member.name.endswith(".tsv"):
        #             with tar.extractfile(member) as f:
        #                 for line in f:
        #                     pid, passage = line.decode("utf-8").strip().split("\t", 1)
        #                     pid = str(pid)
        #                     corpus_test[pid] = passage
        with gzip.open(f"datasets/trec/{year}/msmarco-passagetest20{year}-top1000.tsv.gz", "rt", encoding="utf-8") as f:
            for line in f:
                qid, pid, query, passage = line.strip().split("\t", 3)
                qid = str(qid)
                pid = str(pid)
                if qid not in queries_test:
                    queries_test[qid] = {'query': query,
                                         'passages': {}}
                queries_test[qid]['passages'][pid] = passage
        with open(f"datasets/trec/{year}/20{year}qrels-pass.txt", "r", encoding="utf-8") as f:
            for line in f:
                qid, _, docid, rating = line.strip().split()
                qid = str(qid)
                docid = str(docid)
                rating = int(rating)
                if qid not in qrels_test:
                    qrels_test[qid] = {}
                qrels_test[qid][docid] = rating
    else:
        try:
            corpus_test, queries_test, qrels_test = GenericDataLoader(f"datasets/{dataset}").load(split="test")
        except:
            try:
                corpus_test, queries_test, qrels_test = GenericDataLoader(f"datasets/../datasets/{dataset}").load(split="test")
            except:
                print("Dataset not found")
                exit(1)
        corpus_test = {k: v['text'] for k, v in corpus_test.items()}
    
    return corpus_test, queries_test, qrels_test

# ========================================
# RankLLM utils
# ========================================


def _convert_to_rankllm_request(query: str, passages: Dict[str, str], qid: str) -> 'Request':
    """Convert internal format to RankLLM Request format."""
    query_obj = Query(text=query, qid=qid)
    candidates = []
    for doc_id, text in passages.items():
        candidates.append(Candidate(
            docid=doc_id,
            score=0.0,
            doc={"text": text, "title": ""}
        ))
    return Request(query=query_obj, candidates=candidates)

def _convert_from_rankllm_result(result: 'Result') -> Dict[str, float]:
    """Convert RankLLM Result back to internal format.

    Some RankLLM rerankers return a single Result, others return a list
    (either a list of Result objects or a list of Candidates). This helper
    normalizes all of those shapes into a single score dict.
    """

    # Normalize different return shapes
    if isinstance(result, list):
        if result and hasattr(result[0], "candidates"):
            candidates = []
            for res in result:
                candidates.extend(res.candidates)
        else:
            candidates = result
    else:
        candidates = result.candidates

    reranked_dict = {}
    for rank, candidate in enumerate(candidates):
        # Higher score for higher rank (inverse of position)
        reranked_dict[str(candidate.docid)] = len(candidates) - rank
    return reranked_dict

def _truncate_passages_for_rankllm(query: str, passages: Dict[str, str], reranker, context_size=4096) -> Dict[str, str]:
    reranker_obj = getattr(reranker, "_reranker", None)
    tokenizer_obj = getattr(reranker_obj, "_tokenizer", None) if reranker_obj is not None else None

    if tokenizer_obj is not None and passages:
        # Keep this guard only for Zephyr/Vicuna branch to avoid decoder prompt > 4096.
        target_ctx = context_size
        reserved = 512
        remaining = max(256, target_ctx - reserved)

        query_ids = tokenizer_obj(query, add_special_tokens=False).get("input_ids", [])
        remaining = max(128, remaining - len(query_ids))

        truncated_passages = {}
        for pid, text in passages.items():
            if remaining <= 0:
                break

            p_ids = tokenizer_obj(text, add_special_tokens=False).get("input_ids", [])
            if not p_ids:
                continue

            if len(p_ids) > remaining:
                p_ids = p_ids[:remaining]

            truncated_text = tokenizer_obj.decode(p_ids, skip_special_tokens=True)
            truncated_passages[pid] = truncated_text
            remaining -= len(p_ids)

        if truncated_passages:
            passages = truncated_passages
    return passages


def build_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Build plain prompt text from chat messages for backends without native chat mode."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    system_prompt = messages[0].get("content", "") if messages else ""
    user_prompt = messages[1].get("content", "") if len(messages) > 1 else ""
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def count_prompt_tokens(tokenizer, messages: List[Dict[str, str]]) -> Optional[int]:
    """Return prompt token count when tokenizer is available."""
    if tokenizer is None:
        return None
    try:
        prompt = build_chat_prompt(tokenizer, messages)
        tokenized = tokenizer(prompt, add_special_tokens=False)
        return len(tokenized.get("input_ids", []))
    except Exception:
        return None


def fit_filter_passages_to_context(
    query: str,
    passages: Dict[str, str],
    prompts: Dict[str, str],
    tokenizer,
    context_size: int,
    filter_topk: int,
) -> Dict[str, str]:
    """Shrink filter passages so prompt fits model context window."""
    if not passages or tokenizer is None:
        return passages

    reserved_for_generation = 10 * filter_topk + 128
    prompt_budget = max(512, context_size - reserved_for_generation)
    items = list(passages.items())

    def build_messages(n_items: int) -> List[Dict[str, str]]:
        subset = items[:n_items]
        plist_s = {str(i): text for i, (_, text) in enumerate(subset)}
        return [
            {"role": "system", "content": prompts['S-prompt']},
            {"role": "user", "content":
                f"{prompts['U-prompt-1']}{json.dumps(plist_s)}\n\n"
                f"{prompts['U-prompt-2']}{query}\n\n"
                f"{prompts['U-prompt-3']}"
            }
        ]

    low, high = 1, len(items)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        tok_count = count_prompt_tokens(tokenizer, build_messages(mid))
        if tok_count is not None and tok_count <= prompt_budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    if best == 0:
        best = 1

    if best < len(items):
        print(f"Info: filter prompt too long, reducing passages from {len(items)} to {best} to fit context")

    return dict(items[:best])


def fit_ranking_passages_to_context(
    query: str,
    passages: Dict[str, str],
    prompts: Dict[str, str],
    tokenizer,
    context_size: int,
) -> Dict[str, str]:
    """Shrink ranking passages so prompt fits model context window."""
    if not passages or tokenizer is None:
        return passages

    reserved_for_generation = 10 * len(passages) + 128
    prompt_budget = max(512, context_size - reserved_for_generation)
    items = list(passages.items())

    def build_messages(n_items: int) -> List[Dict[str, str]]:
        subset = dict(items[:n_items])
        return [
            {"role": "system", "content": prompts['rerank-S-prompt']},
            {"role": "user", "content":
                f"{prompts['rerank-U-prompt-1']}{json.dumps(subset)}\n\n"
                f"{prompts['rerank-U-prompt-2']}{query}\n\n"
                f"{prompts['rerank-U-prompt-3']}"
            }
        ]

    low, high = 1, len(items)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        tok_count = count_prompt_tokens(tokenizer, build_messages(mid))
        if tok_count is not None and tok_count <= prompt_budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    if best == 0:
        best = 1

    if best < len(items):
        print(f"Info: ranking prompt too long, reducing passages from {len(items)} to {best} to fit context")

    return dict(items[:best])


import gc
import time

def cleanup_stage_engine(engine, attribute_name='rankllm_reranker', verbose=True, wait_seconds=1.0):
    """
    Best-effort cleanup for FilterStage / RankingStage in all_in_one_rankllm.py.

    This clears:
      - engine.rankllm_reranker / rankllm_filter
      - engine.model
      - engine.pipe
      - engine.tokenizer
      - engine.bert_model
      - engine.bert_tokenizer

    And tries to shutdown nested vLLM-like handlers if present.
    """

    def log(msg):
        if verbose:
            print(msg)

    if engine is None:
        log("Cleanup: engine is None, nothing to do.")
        return

    rr = None

    try:
        # 1) Main RankLLM reranker attribute used by both FilterStage and RankingStage
        rr = getattr(engine, attribute_name, None)

        if rr is not None:
            log(f"Cleanup: found engine.{attribute_name}")

            # Try common shutdown methods directly on the reranker
            for method_name in ("shutdown", "close", "terminate"):
                method = getattr(rr, method_name, None)
                if callable(method):
                    try:
                        method()
                        log(f"Cleanup: called {attribute_name}.{method_name}()")
                    except Exception as e:
                        log(f"Cleanup warning: {attribute_name}.{method_name}() failed: {e}")

            # Try common nested handler names used by custom wrappers / vLLM holders
            for attr_name in (
                "_vllm_handler",
                "vllm_handler",
                "_llm",
                "llm",
                "_engine",
                "engine",
                "_client",
                "client",
            ):
                nested = getattr(rr, attr_name, None)
                if nested is None:
                    continue

                for method_name in ("shutdown", "close", "terminate"):
                    method = getattr(nested, method_name, None)
                    if callable(method):
                        try:
                            method()
                            log(f"Cleanup: called {attribute_name}.{attr_name}.{method_name}()")
                        except Exception as e:
                            log(f"Cleanup warning: {attribute_name}.{attr_name}.{method_name}() failed: {e}")

        # 2) Clear exact heavy attributes present in your script
        for attr_name in (
            attribute_name,
            "model",
            "pipe",
            "tokenizer",
            "bert_model",
            "bert_tokenizer",
        ):
            if hasattr(engine, attr_name):
                try:
                    setattr(engine, attr_name, None)
                    log(f"Cleanup: cleared engine.{attr_name}")
                except Exception as e:
                    log(f"Cleanup warning: could not clear engine.{attr_name}: {e}")

        # Optional short wait to let subprocesses exit
        if wait_seconds and wait_seconds > 0:
            time.sleep(wait_seconds)

    except Exception as e:
        log(f"Cleanup warning: unexpected cleanup error: {e}")

    # 3) Drop local refs
    del rr
    del engine

    # 4) Force Python GC
    collected = gc.collect()
    log(f"Cleanup: gc.collect() -> {collected}")

    # 5) Release CUDA cached memory
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            log("Cleanup: torch.cuda.empty_cache() done")
        except Exception as e:
            log(f"Cleanup warning: torch.cuda.empty_cache() failed: {e}")

        try:
            torch.cuda.ipc_collect()
            log("Cleanup: torch.cuda.ipc_collect() done")
        except Exception as e:
            log(f"Cleanup warning: torch.cuda.ipc_collect() failed: {e}")