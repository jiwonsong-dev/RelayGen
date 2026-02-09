#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import random
import json
from statistics import mean, median, pstdev

# When running this script via `python RouteLRM/measure_latency.py`,
# Python adds this directory to sys.path automatically. The following is
# for safety if invoked from unusual contexts.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import (
    get_dataset,
    get_first_user_msg,
    get_sampling_params,
    relaygen_chat_completions,
    generate_all,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_problem(dataset_name: str, dataset, problem_id: int):
    """Load problem and options based on dataset_name."""
    options = None
    correct_answer = None
    problem_data = None

    if dataset_name == "aime":
        problem = dataset["problem"][problem_id - 60]
    elif dataset_name == "aime25":
        problem = dataset["problem"][problem_id]
    elif dataset_name == "math":
        problem = dataset["problem"][problem_id]
    elif dataset_name == "amc23":
        problem = dataset["question"][problem_id]
    elif dataset_name == "gpqa":
        problem = dataset["Question"][problem_id]
        choices = [
            dataset["Correct Answer"][problem_id],
            dataset["Incorrect Answer 1"][problem_id],
            dataset["Incorrect Answer 2"][problem_id],
            dataset["Incorrect Answer 3"][problem_id],
        ]
        random.seed(time.time())
        random.shuffle(choices)
        options = {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}
        correct_answer = next((k for k, v in options.items() if v == dataset["Correct Answer"][problem_id]), None)
    elif dataset_name == "alpaca_eval":
        problem = dataset[problem_id]["instruction"]
    elif dataset_name == "ifeval":
        problem_data = dataset[problem_id]
        problem = problem_data["prompt"]
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return problem, options, correct_answer, problem_data


def measure_single(dataset_name: str, problem_id: int, model_size: str, budget: int, measure_count: int):
    """Run single model inference measure_count times and collect timing/throughput stats."""
    ds = get_dataset(dataset_name)
    problem, options, _, problem_data = load_problem(dataset_name, ds, problem_id)

    times = []
    throughputs = []
    tokens_list = []

    for i in range(measure_count):
        start = time.time()
        reasoning_answer_str, finished, num_output_tokens = generate_all(
            problem, model_size, budget, options=options, dataset_name=dataset_name
        )
        end = time.time()
        dt = end - start

        # num_output_tokens is returned from generate_all
        tokens = int(num_output_tokens)
        thr = (tokens / dt) if dt > 0 else 0.0
        times.append(dt)
        tokens_list.append(tokens)
        throughputs.append(thr)
        logging.info(f"[Single {i+1}/{measure_count}] time: {dt:.3f}s, throughput: {thr:.2f} tok/s, tokens: {tokens}")

    return {
        "times": times,
        "throughputs": throughputs,
        "tokens": tokens_list,
        "dataset_name": dataset_name,
        "problem_id": problem_id,
        "mode": "single",
        "model_size": model_size,
        "budget": budget,
        "measure_count": measure_count,
    }


def measure_relaygen(dataset_name: str, problem_id: int, small_model: str, base_model: str, budget: int, measure_count: int, verbose: bool):
    """Run relaygen measure_count times and collect timing/throughput stats."""
    ds = get_dataset(dataset_name)
    problem, options, correct_answer, problem_data = load_problem(dataset_name, ds, problem_id)

    user_content = get_first_user_msg(problem, options=options, dataset_name=dataset_name)
    messages = [{"role": "user", "content": user_content}]
    temperature, top_p, top_k, presence_penalty = get_sampling_params(base_model)

    times = []
    throughputs = []
    tokens_list = []
    base_tokens_list = []
    small_tokens_list = []
    switch_counts_list = []

    for i in range(measure_count):
        start = time.time()
        extra_body = {
            "top_k": top_k,
            "presence_penalty": presence_penalty,
        }
        response = relaygen_chat_completions(
            messages=messages,
            base_model_size=base_model,
            small_model_size=small_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=budget,
            extra_body=extra_body,
            verbose=verbose,
        )
        end = time.time()
        dt = end - start

        if isinstance(response, dict) and "error" in response:
            logging.error(f"Relaygen error: {response['error']['message']}")
            tokens = 0
        else:
            usage = response.get("usage", {})
            tokens = int(usage.get("completion_tokens", 0))
            
            relaygen_stats = usage.get("relaygen_stats", {})
            base_tokens_list.append(relaygen_stats.get("base_model_tokens", 0))
            small_tokens_list.append(relaygen_stats.get("small_model_tokens", 0))

        thr = (tokens / dt) if dt > 0 else 0.0
        times.append(dt)
        tokens_list.append(tokens)
        throughputs.append(thr)
        logging.info(f"[RelayGen {i+1}/{measure_count}] time: {dt:.3f}s, throughput: {thr:.2f} tok/s, tokens: {tokens}")

    return {
        "times": times,
        "throughputs": throughputs,
        "tokens": tokens_list,
        "base_tokens": base_tokens_list,
        "small_tokens": small_tokens_list,
        "dataset_name": dataset_name,
        "problem_id": problem_id,
        "mode": "relaygen",
        "small_model": small_model,
        "base_model": base_model,
        "budget": budget,
        "measure_count": measure_count,
    }


def summarize_stats(times, throughputs, tokens, base_tokens=None, small_tokens=None):
    """Compute summary statistics for arrays."""
    def safe_mean(arr):
        return mean(arr) if arr else 0.0
    def safe_median(arr):
        return median(arr) if arr else 0.0
    def safe_std(arr):
        return pstdev(arr) if len(arr) > 1 else 0.0
    def safe_min(arr):
        return min(arr) if arr else 0.0
    def safe_max(arr):
        return max(arr) if arr else 0.0

    def trimmed_mean(arr):
        """Mean excluding a single fastest and slowest value when possible."""
        if len(arr) >= 3:
            arr_sorted = sorted(arr)
            trimmed = arr_sorted[1:-1]
            return mean(trimmed) if trimmed else 0.0
        return mean(arr) if arr else 0.0

    stats = {
        "count": len(times),
        # time stats
        "time_avg": safe_mean(times),
        "time_med": safe_median(times),
        "time_std": safe_std(times),
        "time_min": safe_min(times),
        "time_max": safe_max(times),
        "time_avg_trim": trimmed_mean(times),
        # throughput stats
        "thr_avg": safe_mean(throughputs),
        "thr_med": safe_median(throughputs),
        "thr_std": safe_std(throughputs),
        "thr_min": safe_min(throughputs),
        "thr_max": safe_max(throughputs),
        "thr_avg_trim": trimmed_mean(throughputs),
        # token stats
        "tok_avg": safe_mean(tokens),
        "tok_med": safe_median(tokens),
        "tok_std": safe_std(tokens),
        "tok_min": safe_min(tokens),
        "tok_max": safe_max(tokens),
        "tok_avg_trim": trimmed_mean(tokens),
    }

    if base_tokens:
        stats.update({
            "base_tok_avg": safe_mean(base_tokens),
            "base_tok_med": safe_median(base_tokens),
            "base_tok_min": safe_min(base_tokens),
            "base_tok_max": safe_max(base_tokens),
        })
    
    if small_tokens:
        stats.update({
            "small_tok_avg": safe_mean(small_tokens),
            "small_tok_med": safe_median(small_tokens),
            "small_tok_min": safe_min(small_tokens),
            "small_tok_max": safe_max(small_tokens),
        })

    # Calculate ratios if both exist
    if base_tokens and small_tokens:
        total_avgs = safe_mean(tokens)
        if total_avgs > 0:
            stats["base_ratio_avg"] = safe_mean(base_tokens) / total_avgs
            stats["small_ratio_avg"] = safe_mean(small_tokens) / total_avgs

    return stats


def print_summary(result_dict, stats):
    mode = result_dict["mode"]
    count = result_dict["measure_count"]
    dataset_name = result_dict["dataset_name"]
    problem_id = result_dict["problem_id"]
    budget = result_dict["budget"]
    details = result_dict.get("details", {})

    print("\n" + "=" * 60)
    print(f"Latency Measurement Summary ({mode})")
    print("=" * 60)
    print(f"Dataset: {dataset_name}, Problem ID: {problem_id}, Budget: {budget}")
    if mode == "single":
        print(f"Model: {result_dict['model_size']}")
    else:
        print(f"Base Model: {result_dict['base_model']}, Small Model: {result_dict['small_model']}")
    print(f"Runs: {count}\n")
    
    print(f"Avg Time: {stats['time_avg']:.4f} s (Trimmed: {stats['time_avg_trim']:.4f} s)")
    print(f"Avg Throughput: {stats['thr_avg']:.2f} tok/s (Trimmed: {stats['thr_avg_trim']:.2f} tok/s)")
    print(f"Avg Tokens: {stats['tok_avg']:.1f}")

    print("Time (seconds):")
    print(f"- avg: {stats['time_avg']:.3f}, med: {stats['time_med']:.3f}, std: {stats['time_std']:.3f}, min: {stats['time_min']:.3f}, max: {stats['time_max']:.3f}")
    if stats.get("count", 0) >= 3:
        print(f"- avg (trim, drop fastest & slowest): {stats['time_avg_trim']:.3f}")
    print("Throughput (tokens/s):")
    print(f"- avg: {stats['thr_avg']:.2f}, med: {stats['thr_med']:.2f}, std: {stats['thr_std']:.2f}, min: {stats['thr_min']:.2f}, max: {stats['thr_max']:.2f}")
    if stats.get("count", 0) >= 3:
        print(f"- avg (trim, drop fastest & slowest): {stats['thr_avg_trim']:.2f}")
    print("Tokens (count):")
    print(f"- avg: {stats['tok_avg']:.1f}, med: {stats['tok_med']:.1f}, std: {stats['tok_std']:.1f}, min: {stats['tok_min']:.0f}, max: {stats['tok_max']:.0f}")
    if stats.get("count", 0) >= 3:
        print(f"- avg (trim, drop lowest & highest): {stats['tok_avg_trim']:.1f}")

    if "base_tok_avg" in stats:
        print(f"Avg Base Tokens: {stats['base_tok_avg']:.1f} ({stats.get('base_ratio_avg', 0)*100:.1f}%)")
    if "small_tok_avg" in stats:
        print(f"Avg Small Tokens: {stats['small_tok_avg']:.1f} ({stats.get('small_ratio_avg', 0)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Measure latency for relaygen or single with repeated runs and print stats")
    parser.add_argument("--mode", type=str, choices=["relaygen", "single"], default="single", help="Measurement mode")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "aime25", "math", "gpqa", "amc23", "alpaca_eval", "ifeval"], default="gpqa")
    parser.add_argument("--problem_id", type=int, default=0, help="Problem ID")
    parser.add_argument("--measure_count", type=int, default=3, help="Number of measurements to run")
    parser.add_argument("--budget", type=int, default=32768, help="Token budget")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results as JSON")

    # Single mode args
    parser.add_argument("--model_size", type=str, choices=[
                "qwen3_1.7b", "qwen3_32b",  "wen3_32b_eagle3",
                "r1_1.5b", "r1_32b",], help="Model size for single mode")
    # RelayGen mode args
    parser.add_argument("--small_model_size", type=str, choices=[
                    "qwen3_1.7b","qwen3_32b", "wen3_32b_eagle3",
                    "r1_1.5b", "r1_32b"], help="Small model size for relaygen mode")
    parser.add_argument("--base_model_size", type=str, choices=[
                    "qwen3_1.7b", "qwen3_32b","qwen3_32b_eagle3",
                    "r1_1.5b", "r1_32b"], help="Base model size for relaygen mode")
    
    args = parser.parse_args()

    measure_count = max(int(args.measure_count), 1)

    if args.mode == "single":
        model = args.model_size or "qwen3_32b"
        logging.info(f"Measuring SINGLE with model {model}, budget {args.budget}, runs {measure_count}")
        result = measure_single(args.dataset_name, args.problem_id, model, args.budget, measure_count)
    else:
        small = args.small_model_size or "qwen3_1.7b"
        base = args.base_model_size or "qwen3_32b"
        logging.info(f"Measuring RELAYGEN with small {small}, base {base}, budget {args.budget}, runs {measure_count}")
        result = measure_relaygen(args.dataset_name, args.problem_id, small, base, args.budget, measure_count, args.verbose)

    stats = summarize_stats(
        result["times"], 
        result["throughputs"], 
        result["tokens"],
        base_tokens=result.get("base_tokens"),
        small_tokens=result.get("small_tokens")
    ) 
    print_summary(result, stats)

    if args.output_file:
        output_data = {
            "config": {
                "mode": args.mode,
                "dataset_name": args.dataset_name,
                "problem_id": args.problem_id,
                "measure_count": measure_count,
                "budget": args.budget,
                "model_size": args.model_size if args.mode == "single" else None,
                "small_model_size": args.small_model_size if args.mode == "relaygen" else None,
                "base_model_size": args.base_model_size if args.mode == "relaygen" else None,
            },
            "stats": stats,
            "raw_result": result
        }
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        existing_data = []
        if os.path.exists(args.output_file):
            try:
                with open(args.output_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        existing_data = content
                    else:
                        # If existing file is not a list (e.g. old format), wrap it in a list
                        existing_data = [content]
            except json.JSONDecodeError:
                logging.warning(f"Could not read existing file {args.output_file}, starting fresh.")
                existing_data = []

        existing_data.append(output_data)

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Results appended to {args.output_file} (Total records: {len(existing_data)})")


if __name__ == "__main__":
    main()