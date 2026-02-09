from typing import List, Tuple, Optional, Dict

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.utils import get_first_user_msg


class VLLMOfflineProbMarginCalculator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        chat_template_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.chat_template_path = chat_template_path
        self.chat_template = None

        if chat_template_path:
            try:
                import os
                if os.path.exists(chat_template_path):
                    with open(chat_template_path, "r", encoding="utf-8") as f:
                        self.chat_template = f.read()
            except Exception:
                pass

        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_num_batched_tokens": 512,
            "enforce_eager": True,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        if self.chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception:
                pass
        parts = []
        for m in messages:
            parts.append(f"{m.get('role','user')}: {m.get('content','')}")
        return "\n".join(parts)

    def build_text_to_analyze(self, text, dataset, dataset_name, problem_idx) -> str:
        mp = (self.model_path or "").lower()

        if "aime" in dataset_name and problem_idx >= 60:
            problem_idx -= 60

        if "<think>" not in text:
            text = "\n<think>\n" + text

        if dataset is not None and problem_idx is not None:
            pd = dataset[problem_idx]
            problem = pd.get("problem") or pd.get("Question")
            if dataset_name == "gpqa":
                choices = [
                    pd.get("Correct Answer", ""),
                    pd.get("Incorrect Answer 1", ""),
                    pd.get("Incorrect Answer 2", ""),
                    pd.get("Incorrect Answer 3", ""),
                ]
                options = {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}
                messages = [
                    {"role": "user", "content": get_first_user_msg(problem, options=options)},
                    {"role": "assistant", "content": text},
                ]
            else:
                messages = [
                    {"role": "user", "content": get_first_user_msg(problem)},
                    {"role": "assistant", "content": text},
                ]

            t = self.apply_chat_template(messages, add_generation_prompt=False)
            return t.replace("\n<think>\n\n</think>\n", "")
        return text

    def get_prompt_logprobs_raw(
        self,
        text: str,
        top_logprobs: int = 20,
        dataset=None,
        problem_idx=None,
        dataset_name=None,
        return_tokens: bool = False,
    ):  

        try:
            txt = self.build_text_to_analyze(text, dataset, dataset_name, problem_idx)

            token_ids = self.tokenizer.encode(txt, add_special_tokens=False)
            token_texts = (
                [self.tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
                if return_tokens
                else None
            )
            sp = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                prompt_logprobs=min(max(1, top_logprobs), 20),
                detokenize=False,
            )
            outputs = self.llm.generate([txt], sp)
            if not outputs:
                return ([], token_texts) if return_tokens else []
            req = outputs[0]
            prompt_lps = getattr(req, "prompt_logprobs", None) or []
            return (prompt_lps, token_texts) if return_tokens else prompt_lps
        except Exception:
            return ([], []) if return_tokens else []

    def get_prob_margins(
        self,
        text: str,
        top_logprobs: int = 20,
        dataset=None,
        problem_idx=None,
        dataset_name=None,
    ):
        prompt_lps = self.get_prompt_logprobs_raw(
            text,
            top_logprobs=top_logprobs,
            dataset=dataset,
            problem_idx=problem_idx,
            dataset_name=dataset_name,
            return_tokens=False,
        )
        margins = []
        for lp in prompt_lps:
            vals = [float(getattr(v, "logprob", 0.0)) for v in (lp or {}).values()]
            if len(vals) == 0:
                margins.append(0.0)
            elif len(vals) == 1:
                m1 = float(np.max(vals))
                margins.append(float(np.exp(m1)))
            else:
                s = sorted(vals, reverse=True)
                p1 = np.exp(s[0])
                p2 = np.exp(s[1])
                margins.append(float(p1 - p2))
        return margins
