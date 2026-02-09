

import os
from openai import OpenAI
import logging
import json
import pickle
import random 

from datasets import load_dataset, load_from_disk

# %%
model_names = {
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "qwen3_32b_eagle3": "RedHatAI/Qwen3-32B-speculator.eagle3",
    "r1_1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "r1_32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",

}
ports = {
    "qwen3_1.7b": "30002",
    "qwen3_32b": "30000",
    "qwen3_32b_eagle3": "30001",
    "r1_1.5b": "32001",
    "r1_32b": "32000",
}
clients = {}
for size, full_name in model_names.items():
    clients[size] = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{ports[size]}/v1",
        timeout=3600
    )

THINK_END = ["</think>"]
SENT_END = [".", "!", "?", "\n\n"]

DEFAULT_CUE_TOKENS = [
    "Therefore", "Therefore,", "Thus", "Thus,", "So,", "So ", "Similarly", "Similarly,",
    "Also", "Also,", "Alternatively", "Alternatively,", "Wait", "Wait,", "Hmm", "Hmm,",
    "But", "But,", "However", "However,", "Hence", "Hence,", "Another", "Another,",
    "Oh", "Oh,", "Maybe", "Maybe,", "Other", "Other,", "Again", "Again,", "Now", "Now,",
    "Ah", "Ah,", "Any", "Any,", "Specifically", "Specifically,",
    "therefore", "therefore,", "thus", "thus,", "so,", "so ", "similarly", "similarly,",
    "also", "also,", "alternatively", "alternatively,", "wait", "wait,", "hmm", "hmm,",
    "but", "but,", "however", "however,", "hence", "hence,", "another", "another,",
    "oh", "oh,", "maybe", "maybe,", "other", "other,", "again", "again,", "now", "now,",
    "ah", "ah,", "any", "any,", "specifically", "specifically,",
    "check", "double-check", "verify",
    "then", "then,", "Then", "Then,",
    "next", "next,",  "Next", "Next,",   
]

# Qwen3-1.7B - Qwen3-32B Switch Cues (AMC 2023)
QWEN3_SWITCH_CUES = ["Oh,", "another,", "Thus", "Now", "Alternatively", "alternatively,",
                    "Thus,", "Therefore", "similarly", "similarly,", "now", "Again",
                    "specifically,", "Again,", "Similarly,", "Now,", "Specifically,", "Hence",
                    "Similarly", "Other", "now,", "hence", "Specifically", "So ", 
                    "Therefore,", "Wait,", "Also", "So,"]
    
# R1-1.5B - R1-32B Switch Cues (AMC 2023)
R1_SWITCH_CUES = ["Wait", "Thus", "thus", "similarly", "Again,", "Now",
                    "Therefore", "hence", "Hence,", "Now,", "Thus,", "Oh,",
                    "Similarly,", "Any", "Therefore,", "Alternatively,", "now,", "So,",
                    "now", "verify", "Specifically,", "Alternatively", "Ah,", "wait",
                    "So "]    

def get_model(model_size):
    client = clients[model_size]
    models = client.models.list()
    model = models.data[0].id
    return model

def get_first_user_msg(problem, options=None, instruction=False, multi_turn=False, turn=0, dataset_name=None):
    if options is None: 

        # AIME, MATH
        system_prompt = """
        {problem}
        Please reason step by step, and put your final answer within \boxed{{}}.
        """
        return system_prompt.format(problem=problem)

    else: # GPQA
        system_prompt = """
        Please solve this multiple choice question.

        Question: {problem}.

        Options: 
        A: {ans_a}
        B: {ans_b}
        C: {ans_c}
        D: {ans_d}

        Please provide your answer in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).
        """
        return system_prompt.format(
            problem=problem,
            ans_a=options["A"],
            ans_b=options["B"],
            ans_c=options["C"],
            ans_d=options["D"],
        )


def get_dataset(dataset_name, seed=42):
    if dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name =="aime25":
        dataset = load_dataset("MathArena/aime_2025")["train"]
    elif dataset_name == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "math-train":
        dataset = load_dataset("qwedsacf/competition_math")["train"]
        random.seed(seed)
        indices = random.sample(range(len(dataset)), 1000)
        dataset = dataset.select(indices)
    elif dataset_name == "amc23":
        dataset = load_dataset("math-ai/amc23")["test"]
    elif dataset_name == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    elif dataset_name == "MT-Bench":
        dataset = []
        with open("data/question.jsonl", "r") as ques_file:
            for line in ques_file:
                if line:
                    dataset.append(json.loads(line))
    elif dataset_name == "alpaca_eval": ## Modified only this part
        with open("data/alpaca_eval/alpaca_eval.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        # Filter out 'output' and 'generator' keys from each item in the dataset
        filtered_dataset = [
            {k: v for k, v in item.items() if k not in ("output", "generator")}
            for item in dataset
        ]
        dataset = filtered_dataset
    elif dataset_name == "ifeval":
        with open("data/ifeval.jsonl", "r") as f:
            dataset = [json.loads(line) for line in f]
    else:
        raise NotImplementedError
    return dataset

def get_sampling_params(model_size):

    top_k, presence_penalty = None, None
    if 'qwen' in model_size.lower():
        temperature = 0.6
        top_p = 0.95
        top_k = 20
    elif 'r1' in model_size.lower():
        temperature = 0.6
        top_p = 0.95

    return temperature, top_p, top_k, presence_penalty

def generate_all(problem, model_size, budget=32768, options=None, enable_multiturn=False, turn=0, answers=None, offload=False, dataset_name=None):
    client = clients[model_size]
    
    temperature, top_p, top_k, presence_penalty = get_sampling_params(model_size)

    if not enable_multiturn:
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options, dataset_name=dataset_name)},
        ]
        extra_body = {"add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": True},}
    else:
        messages = []
        for i in range(turn):
            messages.append({"role": "user", "content": get_first_user_msg(problem[i], options, instruction=True, multi_turn=True, turn=i, dataset_name=dataset_name)})
            messages.append({"role": "assistant", "content": f"{answers[i]}"})
        messages.append({"role": "user", "content": get_first_user_msg(problem[turn], options, instruction=True, multi_turn=True, turn=turn, dataset_name=dataset_name)})
        extra_body = {"add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": True},}


    if top_k is not None:
        extra_body["top_k"] = top_k
    if presence_penalty is not None:
        extra_body["presence_penalty"] = presence_penalty

    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=temperature, top_p=top_p,
        max_tokens=budget,
        extra_body=extra_body,
    )

    reasoning_answer_str = response.choices[0].message.content
    finished = "</think>" in reasoning_answer_str

    num_output_tokens = response.usage.completion_tokens

    return reasoning_answer_str, finished, num_output_tokens

def parse_reasoning_and_answer(text):
    if not text:
        return "", ""
    
    # Check for <think> tags first
    if '</think>' in text:
        reasoning_str = text.split('</think>')[0].replace('<think>', '').strip()
        answer_str = text.split('</think>')[1].strip()

        return reasoning_str, answer_str
    
    # If no clear pattern found, treat whole text as reasoning
    return text, ""

def relaygen_chat_completions(messages, base_model_size, small_model_size, 
                                max_tokens=32768, 
                                temperature=None, 
                                top_p=None, 
                                stop=None,
                                extra_body=None,
                                verbose=False,
                                **kwargs
                                ):

    import time
    
    # Extract problem from messages (assume last user message is the problem)
    problem = None

    for msg in reversed(messages):
        if msg.get('role') == 'user':
            problem = msg.get('content')
            break
    
    if not problem:
        raise ValueError("No user message found in messages")

    if "qwen3" in base_model_size:
            SWITCH_CUES = QWEN3_SWITCH_CUES
    elif "r1" in base_model_size:
            SWITCH_CUES = R1_SWITCH_CUES
    else:
        raise NotImplementedError(f"{small_model_size}-{base_model_size} combination is not supporeted.")
    
    # State variables
    mode = "L"  # Start with large model
    sentence_id = 0
    in_think = True

    generated_text = ""
    completion_tokens = 0
    base_tokens = 0
    small_tokens = 0
    
    # Verbose logging setup
    switch_log = []

    
    if verbose:
        print(f"[Starting RelayGen generation [base: {base_model_size}][small: {small_model_size}]")
    
    # Parse switching controls
    eb = extra_body or {}
    

    try:
        while in_think and completion_tokens < max_tokens:
        
            # Determine current model
            current_model = base_model_size if mode == "L" else small_model_size
            
            # Determine stop tokens based on current mode and cues
            stop_tokens = []
            if stop is not None:
                stop_tokens += THINK_END + stop
            else:
                stop_tokens += THINK_END
          
            if mode == "L":
                stop_tokens += SWITCH_CUES
            elif mode == "Ls":
                stop_tokens += SENT_END
            
            remaining = max_tokens - completion_tokens
            chunk_size = remaining
            
            # Create continuation prompt
            continuation_messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": generated_text}
            ]
            
            client = clients[current_model]
                   
            # Set min_tokens only for Large model to prevent "So-loop"
            current_min_tokens = min(5, chunk_size) if mode == "L" else 0

            if generated_text == "":
                temp_extra_body = {
                    "add_generation_prompt": True,
                    "continue_final_message": False,
                    "include_stop_str_in_output": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
                if current_min_tokens > 0:
                    temp_extra_body["min_tokens"] = current_min_tokens
            else:
                temp_extra_body = {
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                    "include_stop_str_in_output": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
                if current_min_tokens > 0:
                    temp_extra_body["min_tokens"] = current_min_tokens
            if eb.get("top_k") is not None:
                temp_extra_body["top_k"] = eb.get("top_k")
            if eb.get("presence_penalty") is not None:
                temp_extra_body["presence_penalty"] = eb.get("presence_penalty")
            
            response = client.chat.completions.create(
                model=get_model(current_model),
                messages=continuation_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=chunk_size,
                stop=stop_tokens,
                extra_body=temp_extra_body,
            )
            
            chunk = response.choices[0].message.content
            chunk_tokens = response.usage.completion_tokens
            finish_reason = response.choices[0].finish_reason

            if not chunk:  # Empty response
                break
            
            # Update token counts and log
            if mode == "L":
                base_tokens += chunk_tokens
                if verbose:
                    print(f"Large model generated {chunk_tokens} tokens | Total: {completion_tokens + chunk_tokens}/{max_tokens} | Large: {base_tokens} | Small: {small_tokens}")
            else:
                small_tokens += chunk_tokens
                if verbose:
                    print(f"Small model generated {chunk_tokens} tokens | Total: {completion_tokens + chunk_tokens}/{max_tokens} | Large: {base_tokens} | Small: {small_tokens}")
            
            # Update generated text and total tokens
            generated_text += chunk
            completion_tokens += chunk_tokens   
            
            switched = False
            # Switching logic
           
            # Check if generation stopped due to SWITCH_CUES (when in large model mode)
            if mode == "L" and finish_reason == "stop":
                for cue in SWITCH_CUES:
                    if chunk.endswith(cue):
                        mode = "Ls"
                        sentence_id += 1
                        switch_log.append({
                            "sentence_id": sentence_id,
                            "switch_type": "L->Ls",
                            "trigger_cues": [cue],
                            "stop_token_triggered": True,
                            "text_len_at_switch": len(generated_text),
                            "tokens_at_switch": completion_tokens
                        })
                        if verbose:
                            print(f"[Switch] L->Ls at sentence {sentence_id}, trigger: {cue} (stop_token) | Tokens - Large: {base_tokens}, Small: {small_tokens}, Total: {completion_tokens}/{max_tokens}")
                        switched = True
                        break
            # Check if generation stopped due to sentence ending (when in small model mode)
            elif mode == "Ls" and finish_reason == "stop":
                for sent_end in SENT_END:
                    if chunk.endswith(sent_end):
                        mode = "L"
                        sentence_id += 1
                        switch_log.append({
                            "sentence_id": sentence_id,
                            "switch_type": "Ls->L",
                            "trigger_cues": ["sentence_end"],
                            "stop_token_triggered": True,
                            "text_len_at_switch": len(generated_text),
                            "tokens_at_switch": completion_tokens
                        })
                        if verbose:
                            print(f"[Switch] Ls->L at sentence {sentence_id}, trigger: sentence_end (stop_token) | Tokens - Large: {base_tokens}, Small: {small_tokens}, Total: {completion_tokens}/{max_tokens}")
                        switched = True
                        break

            # Check for </think> end
            if "</think>" in chunk:
                in_think = False
                break

        if "</think>" not in generated_text:
            generated_text += "\n</think>"
            completion_tokens += 1
        
        if completion_tokens < max_tokens:
            # Answer generation phase (use small model)
            answer_messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": generated_text}
            ]
            
            client = clients[small_model_size]
            
            temp_extra_body = {
                "add_generation_prompt": False,
                "continue_final_message": True,
                "include_stop_str_in_output": True,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            if eb.get("top_k") is not None:
                temp_extra_body["top_k"] = eb.get("top_k")
            if eb.get("presence_penalty") is not None:
                temp_extra_body["presence_penalty"] = eb.get("presence_penalty")
            
            answer_response = client.chat.completions.create(
                model=get_model(small_model_size),
                messages=answer_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens - completion_tokens,
                extra_body=temp_extra_body,
            )
            
            answer_content = answer_response.choices[0].message.content
            answer_tokens = answer_response.usage.completion_tokens
            completion_tokens += answer_tokens
            small_tokens += answer_tokens

            finish_reason = answer_response.choices[0].finish_reason
            
            if verbose:
                print(f"Answer phase: Small model generated {answer_tokens} tokens")
        
            
        else:
            answer_content = ""
            answer_tokens = 0

            finish_reason = "length"

            if verbose:
                print(f"Answer phase: Answer generation skipped due to max tokens")
            
        final_content = generated_text + answer_content
        
        # Calculate switching statistics
        l_to_ls_switches = len([s for s in switch_log if s['switch_type'] == 'L->Ls'])
        ls_to_l_switches = len([s for s in switch_log if s['switch_type'] == 'Ls->L'])
        total_switches = len(switch_log)
        
        # Calculate percentages and other stats
        large_percentage = (base_tokens / completion_tokens * 100) if completion_tokens > 0 else 0
        small_percentage = (small_tokens / completion_tokens * 100) if completion_tokens > 0 else 0
        avg_tokens_per_session = completion_tokens / (total_switches + 1) if total_switches > 0 else completion_tokens
        switch_rate = total_switches / completion_tokens if completion_tokens > 0 else 0
        
        # Verbose logging: final summary
        if verbose:
            print(f"FINAL TOKEN SUMMARY: Total: {completion_tokens}/{max_tokens} ({completion_tokens/max_tokens*100:.1f}%) | Large: {base_tokens} ({large_percentage:.1f}%) | Small: {small_tokens} ({small_percentage:.1f}%) | Switches: {total_switches}")
            print(f"SWITCH SUMMARY: L->Ls: {l_to_ls_switches}, Ls->L: {ls_to_l_switches}, Total: {total_switches}")
            print(f"SWITCHING STATS: Switch rate: {switch_rate:.4f} per token, Avg tokens per session: {avg_tokens_per_session:.1f}")
        
        # Return OpenAI-compatible response structure
        return {
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"relaygen-{base_model_size}-{small_model_size}",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_content
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
                "relaygen_stats": {
                    "base_model_tokens": base_tokens,
                    "small_model_tokens": small_tokens,
                    "switches": sentence_id,
                    "switching_stats": {
                        "total_switches": total_switches,
                        "l_to_ls_switches": l_to_ls_switches,
                        "ls_to_l_switches": ls_to_l_switches,
                        "avg_tokens_per_session": avg_tokens_per_session,
                        "large_model_percentage": large_percentage,
                        "small_model_percentage": small_percentage,
                        "switch_rate": switch_rate,
                    }
                }
            }
        }
        
    except Exception as e:
        # Return error in OpenAI-compatible format
        return {
            "error": {
                "message": str(e),
                "type": "relaygen_error",
                "code": "generation_failed"
            }
        }


def load_reasoning_strings_from_results(result_dir: str = None, num_examples: int = 10, dataset_name: str = "aime25"):
    """
    Load reasoning strings from result files
    
    Args:
        result_dir: Directory containing result files (e.g., path to qwen3_32b folder)
        num_examples: Number of examples to load
        dataset_name: Dataset name ("aime" or "gpqa")
    
    Returns:
        Tuple of (reasoning_strings, dataset, problem_indices)
    """
    import json
    reasoning_strings = []
    dataset = None
    problem_indices = []
    
    if result_dir and os.path.exists(result_dir):
        print(f"Loading reasoning strings from: {result_dir}")
        
        # Load dataset first
        try:
            dataset = get_dataset(dataset_name)
            print(f"Loaded dataset: {dataset_name}")
        except Exception as e:
            print(f"Warning: Could not load dataset {dataset_name}: {e}")
            dataset = None
        
        # Look for pickle files
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                if file.endswith(('.pkl', '.pickle')):
                    file_path = os.path.join(root, file)
                    try:
                        # Extract problem index from file path
                        problem_idx = None
                        try:
                            # Try to extract problem index from file name or path
                            parts = file_path.split('/')
                            for part in parts:
                                if part.isdigit():
                                    problem_idx = int(part)
                                    break
                        except:
                            problem_idx = len(reasoning_strings)  # Use current count as fallback
                        
                        # Load pickle file
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            
                            # Extract reasoning_str from data structure
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        reasoning_str = None
                                        
                                        if 'reasoning_str' in item:
                                            reasoning_str = item['reasoning_str']
                                        elif 'response' in item:
                                            reasoning_str = item['response']
                                        elif 'answer_str' in item:
                                            reasoning_str = item['answer_str']
                                        
                                        if reasoning_str:
                                            reasoning_strings.append(reasoning_str)
                                            problem_indices.append(problem_idx if problem_idx is not None else len(reasoning_strings) - 1)
                                            
                                        if len(reasoning_strings) >= num_examples:
                                            break
                                            
                            elif isinstance(data, dict):
                                reasoning_str = None
                                
                                if 'reasoning_str' in data:
                                    reasoning_str = data['reasoning_str']
                                elif 'response' in data:
                                    reasoning_str = data['response']
                                elif 'answer_str' in data:
                                    reasoning_str = data['answer_str']
                                
                                if reasoning_str:
                                    reasoning_strings.append(reasoning_str)
                                    problem_indices.append(problem_idx if problem_idx is not None else len(reasoning_strings) - 1)
                        
                        if len(reasoning_strings) >= num_examples:
                            break
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}: {e}")
                        continue
            
            if len(reasoning_strings) >= num_examples:
                break
    
    # Truncate to requested number of examples
    reasoning_strings = reasoning_strings[:num_examples]
    problem_indices = problem_indices[:num_examples]
    
    return reasoning_strings, dataset, problem_indices


def load_reasoning_strings_with_repeats_from_results(result_dir: str = None, num_examples: int = 10, dataset_name: str = "aime25", concat_answer: bool = False):
    import json
    reasoning_strings = []
    dataset = None
    problem_indices = []
    repeat_indices = []

    if result_dir and os.path.exists(result_dir):
        try:
            dataset = get_dataset(dataset_name)
        except Exception:
            dataset = None

        entries = []
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                if file.endswith((".pkl", ".pickle")):
                    file_path = os.path.join(root, file)
                    parts = file_path.split("/")
                    digits = []
                    for p in parts:
                        if p.isdigit():
                            digits.append(int(p))
                        else:
                            base, _ = os.path.splitext(p)
                            if base.isdigit():
                                digits.append(int(base))
                    if len(digits) >= 2:
                        pid = digits[-2]
                        rid = digits[-1]
                    elif len(digits) == 1:
                        pid = digits[0]
                        rid = 0
                    else:
                        pid = None
                        rid = 0
                    entries.append((rid, pid, file_path))

        entries.sort(key=lambda x: (x[0] if x[0] is not None else 0, x[1] if x[1] is not None else -1))

        for rid, pid, file_path in entries:
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                rs = None
                if isinstance(data, dict):
                    reasoning = data.get("reasoning_str")
                    answer = data.get("answer_str")
                    response = data.get("response")
                    if concat_answer and reasoning and answer:
                        rs = str(reasoning) + "</think>" + str(answer)
                    elif reasoning is not None:
                        rs = reasoning
                    elif response is not None:
                        rs = response
                    elif answer is not None:
                        rs = answer
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            reasoning = item.get("reasoning_str")
                            answer = item.get("answer_str")
                            response = item.get("response")
                            if concat_answer and reasoning and answer:
                                rs = str(reasoning) + "</think>" + str(answer)
                            elif reasoning is not None:
                                rs = reasoning
                            elif response is not None:
                                rs = response
                            elif answer is not None:
                                rs = answer
                            if rs:
                                break
                if rs:
                    reasoning_strings.append(rs)
                    problem_indices.append(pid if pid is not None else len(reasoning_strings) - 1)
                    repeat_indices.append(rid if rid is not None else 0)
                if len(reasoning_strings) >= num_examples:
                    break
            except Exception:
                continue

    reasoning_strings = reasoning_strings[:num_examples]
    problem_indices = problem_indices[:num_examples]
    repeat_indices = repeat_indices[:num_examples]
    return reasoning_strings, dataset, problem_indices, repeat_indices
