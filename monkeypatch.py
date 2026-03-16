import copy
from typing import Optional

# RUN LLM ASYNC MONKEYPATCH
    
def apply_run_llm_monkeypatch():
    from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM

    if not hasattr(ListwiseRankLLM, "run_llm"):
        print("Warning: Could not monkey patch ListwiseRankLLM: no attribute 'run_llm'")
        return

    og_run_llm = ListwiseRankLLM.run_llm

    def new_run_llm(self, *args, **kwargs):
        try:
            llm_out = og_run_llm(self, *args, **kwargs)
            print(f"\nLLM output: {llm_out}")
            return llm_out
        except Exception as e:
            print(f"Error in run_llm: {e}")
            raise

    ListwiseRankLLM.run_llm = new_run_llm

def apply_run_llm_batched_monkeypatch():
    from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM

    if not hasattr(ListwiseRankLLM, "run_llm_batched"):
        print("Warning: Could not monkey patch ListwiseRankLLM: no attribute 'run_llm_batched'")
        return

    og_run_llm_batched = ListwiseRankLLM.run_llm_batched

    def new_run_llm_batched(self, *args, **kwargs):
        try:
            llm_out = og_run_llm_batched(self, *args, **kwargs)
            print(f"\nLLM batched output: {llm_out}")
            return llm_out
        except Exception as e:
            print(f"Error in run_llm_batched: {e}")
            raise

    ListwiseRankLLM.run_llm_batched = new_run_llm_batched

# FILTER MONKEYPATCH
import os
import random
import torch

from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.vllm_handler import VllmHandler
from rank_llm.rerank.vllm_handler_with_openai_sdk import VllmHandlerWithOpenAISDK
from rank_llm.rerank.rankllm import PromptMode


class FilterListwiseOSLLM(RankListwiseOSLLM):
    def __init__(
        self,
        model: str,
        name: str = "",
        context_size: int = 4096,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: Optional[str] = None,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 100,
        stride: int = 10,
        system_message: Optional[str] = None,
        is_thinking: bool = False,
        reasoning_token_budget: int = 10000,
        use_logits: bool = False,
        use_alpha: bool = False,
        sglang_batched: bool = False,
        tensorrt_batched: bool = False,
        batch_size: int = 32,
        base_url: Optional[str] = None,
        n_keep: int = 10,
        strict_exact_n: bool = True,
        gpu_memory_utilization: float = 0.75,
    ) -> None:
        self.N_KEEP = n_keep
        self.STRICT_EXACT_N = strict_exact_n
        self.GPU_MEMORY_UTILIZATION = gpu_memory_utilization

        if prompt_template_path is None:
            prompt_template_path = f"prompt_templates/filter_listwise_nkeep{self.N_KEEP}.yaml"

        # Initialize only the ListwiseRankLLM base, not RankListwiseOSLLM,
        # to avoid building an uncapped vLLM engine first.
        ListwiseRankLLM.__init__(
            self,
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            window_size=window_size,
            stride=stride,
            use_alpha=use_alpha,
            device=device,
            batch_size=batch_size,
        )

        # Reproduce RankListwiseOSLLM attribute setup
        self._sglang_batched = sglang_batched
        self._tensorrt_batched = tensorrt_batched
        self._name = name
        self._variable_passages = variable_passages
        self._system_message = system_message
        self._is_thinking = is_thinking
        self._reasoning_token_budget = reasoning_token_budget
        self._output_token_estimate = None
        self._use_logits = use_logits
        self._num_gpus = num_gpus
        self._base_url = base_url

        if self._device == "cuda":
            assert torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus

        if prompt_mode and prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. "
                f"The only prompt mode currently supported is a slight variation of {PromptMode.RANK_GPT} prompt."
            )

        if sglang_batched:
            if Engine is None:
                raise ImportError(
                    "Please install rank-llm with `pip install rank-llm[sglang]` to use sglang batch inference."
                )
            assert Engine is not None
            port = random.randint(30000, 35000)
            self._llm = Engine(model, port=port)
            self._tokenizer = self._llm.get_tokenizer()

        elif tensorrt_batched:
            try:
                from tensorrt_llm import LLM as TRTLLM
                from tensorrt_llm import BuildConfig
            except Exception:
                raise ImportError(
                    "Please install rank-llm with `pip install -e .[tensorrt-llm]` to use tensorrt batch inference."
                )
            build_config = BuildConfig(max_seq_len=context_size)
            self._llm = TRTLLM(model=model, build_config=build_config)
            self._tokenizer = self._llm.tokenizer

        else:
            if self._base_url:
                self._vllm_handler = VllmHandlerWithOpenAISDK(
                    model=model,
                    base_url=base_url,
                )
            else:
                print(">>> Building single capped vLLM handler")
                print(">>> max_model_len =", context_size)
                print(">>> gpu_memory_utilization =", self.GPU_MEMORY_UTILIZATION)

                self._vllm_handler = VllmHandler(
                    model=model,
                    download_dir=os.getenv("HF_HOME"),
                    enforce_eager=False,
                    max_logprobs=30,
                    tensor_parallel_size=num_gpus,
                    gpu_memory_utilization=self.GPU_MEMORY_UTILIZATION,
                    trust_remote_code=True,
                    max_model_len=context_size,
                    disable_log_stats=True,
                )

            self._tokenizer = self._vllm_handler.get_tokenizer()

    def receive_permutation(
        self,
        result,
        permutation: str,
        rank_start: int,
        rank_end: int,
        logging: bool = False,
    ):
        """
        Monkeypatch behavior:
        - parse LLM permutation as usual
        - keep ONLY the first N_KEEP valid unique indices from the LLM output
        - return ONLY those N_KEEP candidates (truncate result.candidates)
        """
        # Extract the relevant candidates (same as upstream)
        cut_range = copy.deepcopy(result.candidates[rank_start:rank_end])
        original_rank = list(range(len(cut_range)))

        try:
            # Parse + normalize indices (same as upstream)
            response = self._inference_handler._clean_response(
                permutation, use_alpha=self._use_alpha
            )
            response = [int(x) - 1 for x in response.split()]
            response = self._remove_duplicate(response)
            response = [i for i in response if i in original_rank]  # drop out-of-range

            # Keep ONLY top-N_KEEP from the LLM output
            response = response[:self.N_KEEP]

            if self.STRICT_EXACT_N and len(response) != self.N_KEEP:
                raise ValueError(
                    f"LLM returned {len(response)} valid ids, expected {self.N_KEEP}. "
                    f"Raw permutation: {permutation!r}"
                )

        except Exception as e:
            if logging:
                print(f"exception {e} happened while handling response {permutation!r}")
            # Decide your fallback policy:
            # - strict mode already raised above
            # - in non-strict mode: return empty or keep original top-N slice
            response = [] if self.STRICT_EXACT_N else original_rank[:self.N_KEEP]

        # Build the final selected candidates (ONLY N_KEEP, no backfill)
        selected_candidates = [copy.deepcopy(cut_range[i]) for i in response]

        # If you want to preserve scores from the selected docs, they are already part of the deep-copied objects.
        # (Avoid reassigning score from cut_range[j] like upstream, which can mismatch after reordering.)

        # Truncate the entire result to ONLY those N candidates
        result.candidates = selected_candidates
        return result
    
# RANKOSLLM CAPPED
import os
from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM
from rank_llm.rerank.vllm_handler import VllmHandler

class RankListwiseOSLLM_Capped(RankListwiseOSLLM):
    def __init__(self, *args, gpu_memory_utilization=0.75, **kwargs):
        super().__init__(*args, **kwargs)

        # Rebuild vLLM engine with explicit limits (only for the local vLLM path)
        if hasattr(self, "_vllm_handler") and not getattr(self, "_base_url", None):
            self._vllm_handler = VllmHandler(
                model=self._model,
                download_dir=os.getenv("HF_HOME"),
                enforce_eager=False,
                max_logprobs=30,
                tensor_parallel_size=self._num_gpus,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=self._context_size,   # <- this is the key
            )
            self._tokenizer = self._vllm_handler.get_tokenizer()