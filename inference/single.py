# %%
import os
import sys
import pickle
import pprint
import logging
import argparse
import random
import time
import json

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.utils import get_dataset, generate_all

# %%
parser = argparse.ArgumentParser(description="Runs speculative reasoning using a small model")
parser.add_argument("--dataset_name", type=str, choices=["aime", "aime25", "math", "math-train", "gpqa", "amc23", "alpaca_eval", "ifeval"], default="gpqa",
                    help="Dataset")
parser.add_argument("--seed", type=int, default=42,
                    help="Random Seed")
parser.add_argument("--model_size", type=str, choices=["qwen3_1.7b", "qwen3_32b", "qwen3_32b_eagle3",
                                                        "r1_1.5b", "r1_32b"])
parser.add_argument("--budget", type=int, default=32768,
                    help="Max num of total output tokens in each step")
# problem_id: 60-89 for AIME, 0-99 for math, 0-99 for GPQA
parser.add_argument("--problem_id", type=int, default=0,
                    help="Query ID")
parser.add_argument("--repeat_id", type=int, default=0,
                    help="Repeat ID (0-15, k=16)")
parser.add_argument("--output_dir", type=str, default="outputs", 
                    help="Where result pickle files will be written to")
parser.add_argument("--verbose", action="store_true",
                    help="Enable verbose logging")
parser.add_argument("--measure_count", type=int, default=1,
                    help="Number of speed measurements to run and average")
args, _ = parser.parse_known_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


args.dataset = get_dataset(args.dataset_name)

# %%
if args.dataset_name == "aime":
    problem = args.dataset["problem"][args.problem_id - 60]
    options = None
elif args.dataset_name == "aime25":
    problem = args.dataset["problem"][args.problem_id]
    options = None
elif args.dataset_name == "math":
    problem = args.dataset["problem"][args.problem_id]
    options = None
elif args.dataset_name == "math-train":
    problem= args.dataset["problem"][args.problem_id]
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

problem_id = f"{args.dataset_name}_{args.problem_id}"

# Output filename
output_filename = os.path.join(args.output_dir, f"{args.problem_id}/{args.repeat_id}")
if os.path.exists(f"{output_filename}.pickle"):
    logging.info(f"Problem {args.problem_id} repeat {args.repeat_id} resolved, exiting")
    exit()
    

metadata_list = []
try:
    logging.info(f"[{problem_id}] Single model generation")
    logging.info(f"Model: {args.model_size}")
    logging.info(f"Budget: {args.budget} tokens")

    measure_count = max(int(args.measure_count), 1)
    times = []
    tokens_list = []
    throughputs = []
    last_metadata = None

    for i in range(measure_count):
        start_time = time.time()
        reasoning_answer_str, finished, num_output_tokens = generate_all(problem, args.model_size, args.budget, options=options, dataset_name=args.dataset_name)
        end_time = time.time()
        generation_time = end_time - start_time

        if finished:

            reasoning_str = reasoning_answer_str.split('</think>')[0]
            answer_str = reasoning_answer_str.split('</think>')[1]
        else:
            reasoning_str = reasoning_answer_str
            answer_str = None

        metadata = {
            "reasoning_str": reasoning_str,
            "answer_str": answer_str,
            "num_output_tokens": num_output_tokens,
            "generation_time": generation_time
        }

        if args.dataset_name == "alpaca_eval":
            metadata["dataset"] = args.dataset[args.problem_id]["dataset"]
            metadata["instruction"] = args.dataset[args.problem_id]["instruction"]
        elif args.dataset_name == "gpqa":
            metadata["correct_answer"] = correct_answer
        elif args.dataset_name == "ifeval":
            metadata["prompt"] = problem_data["prompt"]
            metadata["instruction_id_list"] = problem_data["instruction_id_list"]
            metadata["kwargs"] = problem_data["kwargs"]
            metadata["gen"] = [reasoning_answer_str]

        last_metadata = metadata
        t = generation_time
        tok = int(num_output_tokens)
        thr = (tok / t) if t > 0 else 0.0
        times.append(t)
        tokens_list.append(tok)
        throughputs.append(thr)

    avg_time = sum(times) / len(times) if times else 0.0
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0

    if last_metadata:
        last_metadata["avg_generation_time"] = avg_time
        last_metadata["avg_throughput"] = avg_throughput
        last_metadata["measure_count"] = measure_count
        metadata_list.append(last_metadata)
    
except ValueError:
    logging.error(f"ValueError caught in chat template application, continuing")

os.makedirs(os.path.dirname(f"{output_filename}.pickle"), exist_ok=True)

with open(f"{output_filename}.pickle", "wb") as f:
    pickle.dump(metadata_list, f)

with open(f"{output_filename}.txt", "w") as f:
    pprint.pprint(metadata_list, stream=f)

if args.verbose:
    logging.info(f"Results saved to {output_filename}")

