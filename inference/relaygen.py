#!/usr/bin/env python3

import os
import pickle
import pprint
import logging
import argparse
import random
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_dataset, get_first_user_msg, get_sampling_params, relaygen_chat_completions, parse_reasoning_and_answer
import re

# Argument parsing
parser = argparse.ArgumentParser(description="Runs RelayGen with dynamic model switching using single wrapped function call")
parser.add_argument("--dataset_name", type=str, choices=["aime", "aime25", "math", "gpqa", "ifeval"], default="gpqa",
                    help="Dataset")
parser.add_argument("--seed", type=int, default=42,
                    help="Random Seed")
parser.add_argument("--small_model_size", type=str, choices=["qwen3_1.7b", "qwen3_32b","qwen3_32b_eagle3",
                                                            "r1_1.5b", "r1_32b",], default="qwen3_1.7b")
parser.add_argument("--base_model_size", type=str, choices=["qwen3_1.7b", "qwen3_32b", "qwen3_32b_eagle3",
                                                            "r1_1.5b", "r1_32b"], default="qwen3_32b")
parser.add_argument("--answer_with_small_model", action="store_true",
                    help="Use small model instead of target model to compose answer")
parser.add_argument("--budget", type=int, default=8192,
                    help="Max num of total output tokens in each step")
parser.add_argument("--problem_id", type=int, default=0,
                    help="Query ID")
parser.add_argument("--repeat_id", type=int, default=0,
                    help="Repeat ID (0-15, k=16)")
parser.add_argument("--output_dir", type=str, default="./output", 
                    help="Where result pickle files will be written to")
parser.add_argument("--verbose", action="store_true",
                    help="Enable verbose logging")
parser.add_argument("--measure_count", type=int, default=1,
                    help="Number of speed measurements to run and average")

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.dataset_name == "ifeval":
    output_filename = os.path.join(args.output_dir, f"ifeval/{args.problem_id}/{args.repeat_id}")
else:
    output_filename = os.path.join(args.output_dir, f"{args.problem_id}/{args.repeat_id}")

if os.path.exists(f"{output_filename}.pickle"):
    logging.info(f"Problem {args.problem_id} repeat {args.repeat_id} resolved, exiting")
    exit()

args.dataset = get_dataset(args.dataset_name)

# Problem setup based on dataset
if args.dataset_name == "aime":
    problem = args.dataset["problem"][args.problem_id - 60]
    options = None
elif args.dataset_name == "aime25":
    problem = args.dataset["problem"][args.problem_id]
    options = None
elif args.dataset_name == "math":
    problem = args.dataset["problem"][args.problem_id]
    options = None
elif args.dataset_name == "amc23":
    problem = args.dataset["question"][args.problem_id]
    options = None
elif args.dataset_name == "gpqa":
    problem = args.dataset["Question"][args.problem_id]
    options = {
        "A": args.dataset["Correct Answer"][args.problem_id],
        "B": args.dataset["Incorrect Answer 1"][args.problem_id],
        "C": args.dataset["Incorrect Answer 2"][args.problem_id],
        "D": args.dataset["Incorrect Answer 3"][args.problem_id],
    }
    correct_answer = "A"
elif args.dataset_name == "alpaca_eval":
    problem = args.dataset[args.problem_id]["instruction"]
    options = None
elif args.dataset_name == "ifeval":
    problem = args.dataset[args.problem_id]["prompt"]
    options = None

problem_id = f"{args.dataset_name}_{args.problem_id}"

def relaygen_generate(problem, base_model_size, small_model_size, budget, options=None, 
                                temperature=0.6, top_p=0.95, top_k=None, presence_penalty=None, 
                                correct_answer=None, dataset_name=None, verbose=False,):

    
    # Prepare the message
    user_content = get_first_user_msg(problem, options=options, dataset_name=dataset_name)
    messages = [{"role": "user", "content": user_content}]
    
    logging.info(f"[{problem_id}] RelayGen generation")
    logging.info(f"Base model: {base_model_size}, Small model: {small_model_size}")
    logging.info(f"Budget: {budget} tokens")
    
    start_time = time.time()
    
    # Single API call to relaygen_chat_completions
    try:
        extra_body = {
            "top_k": top_k,
            "presence_penalty": presence_penalty,
        }

        response = relaygen_chat_completions(
            messages=messages,
            base_model_size=base_model_size,
            small_model_size=small_model_size,
            temperature=temperature,
            top_p=top_p,
            max_tokens=budget,
            extra_body=extra_body,
            verbose=verbose
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check if response contains error
        if 'error' in response:
            raise Exception(response['error']['message'])
        
        # Extract response content
        generated_text = response['choices'][0]['message']['content']
        
        # Extract token usage information
        usage = response.get('usage', {})
        total_tokens = usage.get('completion_tokens', 0)
        relaygen_stats = usage.get('relaygen_stats', {})
        base_tokens = relaygen_stats.get('base_model_tokens', 0)
        small_tokens = relaygen_stats.get('small_model_tokens', 0)
        switching_stats = relaygen_stats.get('switching_stats', {})
        
        logging.info(f"Generation completed in {total_time:.2f} seconds")
        logging.info(f"Total tokens: {total_tokens}")
        logging.info(f"Base model tokens: {base_tokens}")
        logging.info(f"Small model tokens: {small_tokens}")
        
        if switching_stats:
            logging.info(f"Switching statistics: Total switches: {switching_stats.get('total_switches', 0)}, Switch rate: {switching_stats.get('switch_rate', 0):.4f}")
        
        # Parse reasoning and answer from generated text
        reasoning_str, answer_str = parse_reasoning_and_answer(generated_text)
        
        # Return metadata in the same format as original with switching stats
        metadata = {
            "problem_id": problem_id,
            "base_model_size": base_model_size,
            "small_model_size": small_model_size,
            "budget": budget,
            "generation_time": total_time,
            "partial_text": generated_text,
            "reasoning_str": reasoning_str,
            "answer_str": answer_str,
            "total_tokens": total_tokens,
            "base_tokens": base_tokens,
            "small_tokens": small_tokens
        }
        
        # Add switching statistics if available
        if switching_stats:
            metadata["switching_stats"] = switching_stats
        
        # Add correct answer for GPQA cases
        if correct_answer is not None:
            metadata["correct_answer"] = correct_answer
        
        return [metadata]
        
    except Exception as e:
        logging.error(f"Error during RelayGen generation: {str(e)}")
        
        # Create error metadata with correct_answer if available
        error_metadata = {
            "problem_id": problem_id,
            "base_model_size": base_model_size,
            "small_model_size": small_model_size,
            "budget": budget,
            "error": str(e),
            "partial_text": "",
            "reasoning_str": "",
            "answer_str": "",
            "total_tokens": 0,
            "base_tokens": 0,
            "small_tokens": 0,
        }
        
        # Add correct answer for GPQA cases in error case too
        if correct_answer is not None:
            error_metadata["correct_answer"] = correct_answer
        
        return [error_metadata]

# Main execution
# Pass correct_answer for GPQA cases
correct_answer_param = None
if args.dataset_name == "gpqa":
    correct_answer_param = correct_answer

temperature, top_p, top_k, presence_penalty = get_sampling_params(args.base_model_size)
print(f"[Sampling parameters] temperature: {temperature}, top_p: {top_p}, top_k: {top_k}, presence_penalty: {presence_penalty}")

# Clamp measure_count to at least 1
measure_count = max(int(args.measure_count), 1)

times = []
throughputs = []
tokens_list = []
last_metadata = None

for i in range(measure_count):
    run_metadata_list = relaygen_generate(
        problem,
        args.base_model_size,
        args.small_model_size,
        args.budget,
        options,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        correct_answer=correct_answer_param,
        dataset_name=args.dataset_name,
        verbose=args.verbose
    )
    m = run_metadata_list[-1] if run_metadata_list else {}
    last_metadata = m
    t = float(m.get("generation_time", 0.0))
    tok = int(m.get("total_tokens", m.get("num_output_tokens", 0) or 0))
    thr = (tok / t) if t > 0 else 0.0
    times.append(t)
    tokens_list.append(tok)
    throughputs.append(thr)
    logging.info(f"[Measure {i+1}/{measure_count}] time: {t:.3f}s, throughput: {thr:.2f} tok/s, tokens: {tok}")

# Compute averages
avg_time = sum(times) / len(times) if times else 0.0
avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0
logging.info(f"[Average over {measure_count}] time: {avg_time:.3f}s, throughput: {avg_throughput:.2f} tok/s")

# Prepare metadata_list for saving (keep last run metadata and annotate averages)
metadata_list = [last_metadata] if last_metadata else []

# Annotate averages
if metadata_list:
    metadata_list[-1]["avg_generation_time"] = avg_time
    metadata_list[-1]["avg_throughput"] = avg_throughput
    metadata_list[-1]["measure_count"] = measure_count

# Save results
os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)

with open(f"{output_filename}.pickle", "wb") as f:
    pickle.dump(metadata_list, f)

with open(f"{output_filename}.txt", "w") as f:
    pprint.pprint(metadata_list, stream=f)

if args.verbose:
    logging.info(f"Results saved to {output_filename}")
