#!/usr/bin/env python3
"""
Score evaluation script for RelayGen (relay generation) outputs.
This script evaluates the performance of relay generation models on various datasets.
"""

import os
import ast
import argparse
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union

from utils_score.parser import extract_answer
from utils_score.grader import math_equal
from utils.utils import get_dataset


class RelayGenScorer:
    """Scorer for RelayGen model outputs."""
    
    KNOWN_DATASETS = ["aime25", "aime", "math", "amc23", "gpqa"]
    
    def __init__(self, dataset_name: str, root_dir: str, target_num: int = 500):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.target_num = target_num
        self.root_path = self._find_dataset_folder()
        self.dataset = get_dataset(self.dataset_name)
        
    def _find_dataset_folder(self) -> str:
        """Find the appropriate dataset folder in root_dir."""
        available_folders = os.listdir(self.root_dir)
        matching_folder = None
        
        # First, try to find exact or best match for the specified dataset_name
        for folder in available_folders:
            if self.dataset_name in folder:
                matching_folder = folder
                # Check if we need to update to a more specific dataset name
                for dataset in self.KNOWN_DATASETS:
                    if (dataset in folder and dataset != self.dataset_name and 
                        len(dataset) > len(self.dataset_name) and 
                        self.dataset_name in dataset):
                        self.dataset_name = dataset
                        print(f"📝 Updated dataset_name to more specific: {dataset} based on folder: {folder}")
                        break
                break
        
        # If no exact match, try to find any folder that contains a known dataset name
        if matching_folder is None:
            for folder in available_folders:
                for dataset in self.KNOWN_DATASETS:
                    if dataset in folder:
                        matching_folder = folder
                        self.dataset_name = dataset
                        print(f"🔍 Auto-detected dataset: {dataset} from folder: {folder}")
                        break
                if matching_folder:
                    break
        
        # Return the appropriate path
        if matching_folder is None:
            return os.path.join(self.root_dir, self.dataset_name)
        else:
            return os.path.join(self.root_dir, matching_folder)
    
    def _extract_output_data(self, data: List[Dict]) -> Tuple[Optional[str], int, Dict]:
        """Extract output text, generation length, and switching stats from data."""
        last_entry = data[-1]
        
        # Extract output text
        output = None
        if 'final_text' in last_entry:
            output = last_entry['final_text']
        elif 'answer_str' in last_entry:
            output = last_entry['answer_str']
        
        # Extract generation length
        gen_len = 0
        if 'total_tokens' in last_entry:
            gen_len = last_entry['total_tokens']
        elif 'num_output_tokens' in last_entry:
            gen_len = last_entry['num_output_tokens']
        
        # Extract switching statistics
        switching_stats = {}
        if ('switching_stats' in last_entry and 
            last_entry['switching_stats'] is not None and 
            last_entry['switching_stats'] and
            'large_model_percentage' in last_entry['switching_stats']):
            switching_stats = last_entry['switching_stats']
        
        return output, gen_len, switching_stats
    
    def _get_ground_truth_answer(self, problem_id: int, data: List[Dict]) -> str:
        """Get ground truth answer based on dataset type."""
        if self.dataset_name == 'aime':
            return self.dataset['answer'][problem_id - 60]
        elif self.dataset_name == 'aime25':
            return str(self.dataset['answer'][problem_id])
        elif self.dataset_name == 'math':
            return extract_answer(self.dataset['solution'][problem_id], 'math-oai', use_last_number=False)
        elif self.dataset_name == "amc23":
            return str(self.dataset['answer'][problem_id])
        elif self.dataset_name == "gpqa":
            return data[-1]['correct_answer']
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _evaluate_answer(self, output: Optional[str], gt_answer: str) -> bool:
        """Evaluate if the predicted answer matches the ground truth."""
        if output is None:  # budget overflow
            return False
        
        pred_answer = extract_answer(output, self.dataset_name, use_last_number=False)
        return math_equal(pred_answer, gt_answer, timeout=False)
    
    def _count_tokens(self, text: Optional[str]) -> int:
        """Best-effort token count: try tiktoken, otherwise whitespace split length."""
        if not text:
            return 0
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Fallback approximate token count
            return len(text.split())

    def _extract_answer_only(self, output: Optional[str], data: List[Dict]) -> Optional[str]:
        """Prefer explicit answer_str if available, otherwise extract from output."""
        last_entry = data[-1]
        if 'answer_str' in last_entry and last_entry['answer_str']:
            return last_entry['answer_str']
        if output:
            try:
                return extract_answer(output, self.dataset_name, use_last_number=False)
            except Exception:
                return None
        return None
    
    def score_relaygen(self, verbose: bool = True, repeat_id: Optional[Union[int, List[int]]] = None) -> Dict[int, Dict[str, List]]:
        """Main scoring function for RelayGen outputs."""
        print(f"🎯 Evaluating RelayGen outputs for {self.dataset_name}")
        print(f"📂 Root path: {self.root_path}")
        
        target_repeat_ids = None
        if repeat_id is not None:
            if isinstance(repeat_id, int):
                target_repeat_ids = {repeat_id}
            else:
                target_repeat_ids = set(repeat_id)
            print(f"🔄 Filtering for repeat_ids: {sorted(list(target_repeat_ids))}")
        
        # Initialize counters
        total_responses = 0
        total_len = 0
        got_rights = 0
        
        # Statistics tracking
        switching_data = []
        has_switching_stats = False
        metadata_dict = {}
        
        # Get analysis targets (sorted problem folders)
        analysis_targets = sorted(os.listdir(self.root_path), key=int)[:self.target_num]
        
        print(f"📊 Processing {len(analysis_targets)} problems...")
        
        for subfolder_name in analysis_targets:
            subfolder_path = os.path.join(self.root_path, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                continue
                
            problem_id = int(subfolder_name)
            metadata_dict[problem_id] = {
                'correctness': [],
                'gen_len': []
            }
            
            # Process all pickle files in the problem folder
            pickle_files = [f for f in os.listdir(subfolder_path) if f.endswith(".pickle")]
            
            for filename in sorted(pickle_files, key=lambda x: int(os.path.splitext(x)[0])):
                # Filter by repeat_id if specified
                current_repeat_id = int(os.path.splitext(filename)[0])
                if target_repeat_ids is not None and current_repeat_id not in target_repeat_ids:
                    continue

                file_path = os.path.join(subfolder_path, filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    total_responses += 1
                    
                    # Extract data from the pickle file
                    output, gen_len, switching_stats = self._extract_output_data(data)
                    answer_only = self._extract_answer_only(output, data)
                    
                    total_len += gen_len
                    metadata_dict[problem_id]['gen_len'].append(gen_len)
                    
                    # Collect switching statistics if available
                    if switching_stats:
                        has_switching_stats = True
                        # Enrich switching stats with offloaded token counts when possible
                        offloaded_total_tokens = None
                        # Per user definition: "offloaded tokens" = tokens generated by the SMALL model
                        if 'offloaded_tokens' in switching_stats:
                            offloaded_total_tokens = switching_stats.get('offloaded_tokens')
                        elif 'small_model_token_count' in switching_stats:
                            offloaded_total_tokens = switching_stats.get('small_model_token_count')
                        elif 'small_model_percentage' in switching_stats and gen_len:
                            try:
                                offloaded_total_tokens = int(round(gen_len * switching_stats['small_model_percentage'] / 100))
                            except Exception:
                                offloaded_total_tokens = None
                        elif 'large_model_token_count' in switching_stats and gen_len:
                            # Fallback: infer small-model tokens if only large-model count is provided
                            try:
                                offloaded_total_tokens = max(int(gen_len) - int(switching_stats['large_model_token_count']), 0)
                            except Exception:
                                offloaded_total_tokens = None
                        elif 'large_model_percentage' in switching_stats and gen_len:
                            # Fallback: infer small-model percentage from large-model percentage
                            try:
                                offloaded_total_tokens = int(round(gen_len * (100 - switching_stats['large_model_percentage']) / 100))
                            except Exception:
                                offloaded_total_tokens = None
                        enriched = dict(switching_stats)
                        enriched['total_tokens'] = gen_len
                        
                        
                        # Use exact token counts from pickle if available
                        last_entry = data[-1]
                        if 'base_tokens' in last_entry:
                            enriched['base_tokens'] = last_entry['base_tokens']
                        if 'small_tokens' in last_entry:
                            enriched['small_tokens'] = last_entry['small_tokens']
                            # Update offloaded_total_tokens if we have exact count
                            offloaded_total_tokens = last_entry['small_tokens']
                        
                        if offloaded_total_tokens is not None:
                            ans_tok = self._count_tokens(answer_only)
                            # User rule: answer_str is fully offloaded (i.e., counted toward small-model tokens)
                            answer_offloaded = ans_tok
                            # reasoning portion = total_offloaded - answer_str_len (clamped at 0)
                            reasoning_offloaded = max(offloaded_total_tokens - ans_tok, 0)
                            reasoning_tok = max(gen_len - ans_tok, 0)
                            enriched['offloaded_total_tokens'] = offloaded_total_tokens
                            enriched['answer_offloaded_tokens'] = answer_offloaded
                            enriched['reasoning_offloaded_tokens'] = reasoning_offloaded
                            enriched['answer_tokens'] = ans_tok
                            enriched['reasoning_tokens'] = reasoning_tok
                            enriched['answer_offload_ratio'] = (answer_offloaded / ans_tok) if ans_tok > 0 else None
                            enriched['reasoning_offload_ratio'] = (reasoning_offloaded / reasoning_tok) if reasoning_tok > 0 else None
                        switching_data.append(enriched)
                    
                    # Evaluate correctness
                    gt_answer = self._get_ground_truth_answer(problem_id, data)
                    is_correct = self._evaluate_answer(output, gt_answer)
                    
                    if is_correct:
                        got_rights += 1
                        metadata_dict[problem_id]['correctness'].append(1)
                    else:
                        metadata_dict[problem_id]['correctness'].append(0)
                        
                except Exception as e:
                    print(f"⚠️  Error processing {file_path}: {e}")
                    continue
        
        # Always print results, but control per-problem details with verbose
        self._print_results(metadata_dict, total_responses, got_rights, total_len, 
                          has_switching_stats, switching_data, verbose)
        
        return metadata_dict
    
    def _print_results(self, metadata_dict: Dict, total_responses: int, got_rights: int, 
                      total_len: int, has_switching_stats: bool, switching_data: List[Dict], verbose: bool = True):
        """Print evaluation results in a beautiful format."""
        print("\n" + "=" * 60)
        print("🎯 RELAYGEN EVALUATION RESULTS")
        print("=" * 60)
        print(f"📁 Root directory: {self.root_dir}")
        print(f"📊 Dataset: {self.dataset_name.upper()}")
        
        # Calculate Pass@1 metrics
        problem_accuracies = []
        for problem_id, data in metadata_dict.items():
            if data['correctness']:  # Only include problems with responses
                problem_accuracy = sum(data['correctness']) / len(data['correctness'])
                problem_accuracies.append(problem_accuracy)
        
        problem_level_pass_at_1 = sum(problem_accuracies) / len(problem_accuracies) if problem_accuracies else 0
        sample_level_pass_at_1 = got_rights / total_responses if total_responses > 0 else 0
        sample_level_avg_length = total_len / total_responses if total_responses > 0 else 0
        
        # Calculate problem-level average response length
        problem_level_lengths = []
        for problem_id, data in metadata_dict.items():
            if data['gen_len']:  # Only if there are results
                problem_avg_len = sum(data['gen_len']) / len(data['gen_len'])
                problem_level_lengths.append(problem_avg_len)
        problem_level_avg_length = sum(problem_level_lengths) / len(problem_level_lengths) if problem_level_lengths else 0
        
        print(f"\n🎯 Pass@1 (sample-level): {sample_level_pass_at_1:.4f}")
        print(f"🎯 Pass@1 (problem-level): {problem_level_pass_at_1:.4f}")
        print(f"📊 Total responses: {total_responses}")
        print(f"📏 Response length (sample-level avg): {sample_level_avg_length:.2f} tokens")
        print(f"📏 Response length (problem-level avg): {problem_level_avg_length:.2f} tokens")
        
        # Display switching statistics if available
        if has_switching_stats and switching_data:
            self._print_switching_stats(switching_data)
        
        # Display per-problem statistics only if verbose
        if verbose:
            print("\n" + "-" * 60)
            print("📋 PER-PROBLEM STATISTICS")
            print("-" * 60)
            
            for key, value in sorted(metadata_dict.items()):
                if value['correctness']:  # Only if there are results
                    acc = sum(value['correctness']) / len(value['correctness'])
                    gen_len_mean = sum(value['gen_len']) / len(value['gen_len'])
                    print(f"Problem {key:3d}: {len(value['correctness']):2d} repeats, "
                          f"avg_len={gen_len_mean:6.1f}, acc={acc:.3f}")
    
    def _print_switching_stats(self, switching_data: List[Dict]):
        """Print switching statistics for RelayGen."""
        print("\n" + "-" * 60)
        print("🔄 RELAYGEN SWITCHING STATISTICS")
        print("-" * 60)
        
        avg_large_percentage = sum(stats['large_model_percentage'] for stats in switching_data if 'large_model_percentage' in stats) / max(1, len([1 for s in switching_data if 'large_model_percentage' in s]))
        avg_small_percentage = sum(stats['small_model_percentage'] for stats in switching_data if 'small_model_percentage' in stats) / max(1, len([1 for s in switching_data if 'small_model_percentage' in s]))
        avg_switches = sum(stats['total_switches'] for stats in switching_data if 'total_switches' in stats) / max(1, len([1 for s in switching_data if 'total_switches' in s]))
        avg_switch_rate = sum(stats['switch_rate'] for stats in switching_data if 'switch_rate' in stats) / max(1, len([1 for s in switching_data if 'switch_rate' in s]))
        

        print(f"🔵 Average Base Model Generation: {avg_large_percentage:.2f}%")
        print(f"🟢 Average Small Model Generation: {avg_small_percentage:.2f}%")
        print(f"🔄 Average Total Switches: {avg_switches:.1f}")
        print(f"📊 Average Switch Rate: {avg_switch_rate:.4f}")

        # Calculate base and small tokens
        base_totals = []
        small_totals = []
        for s in switching_data:
            # Prefer explicit token counts if available
            if 'base_tokens' in s:
                base_totals.append(s['base_tokens'])
            elif 'large_model_token_count' in s:
                base_totals.append(s['large_model_token_count'])
            elif 'total_tokens' in s and 'offloaded_total_tokens' in s:
                base_totals.append(max(0, s['total_tokens'] - s['offloaded_total_tokens']))
                
            if 'small_tokens' in s:
                small_totals.append(s['small_tokens'])
            elif 'offloaded_total_tokens' in s:
                small_totals.append(s['offloaded_total_tokens'])
        
        if base_totals:
            avg_base_total = sum(base_totals) / len(base_totals)
            print(f"🧮 Average Base Tokens: {avg_base_total:.1f}")
            
        if small_totals:
            avg_small_total = sum(small_totals) / len(small_totals)
            print(f"🧮 Average Small Tokens: {avg_small_total:.1f}")
        
        # Offloaded token statistics (if available or computable)
        off_totals = [s['offloaded_total_tokens'] for s in switching_data if 'offloaded_total_tokens' in s]
        ans_off = [s['answer_offloaded_tokens'] for s in switching_data if 'answer_offloaded_tokens' in s]
        reas_off = [s['reasoning_offloaded_tokens'] for s in switching_data if 'reasoning_offloaded_tokens' in s]
        ans_ratios = [s['answer_offload_ratio'] for s in switching_data if 'answer_offload_ratio' in s and s['answer_offload_ratio'] is not None]
        reas_ratios = [s['reasoning_offload_ratio'] for s in switching_data if 'reasoning_offload_ratio' in s and s['reasoning_offload_ratio'] is not None]
        if off_totals:
            avg_off_total = sum(off_totals) / len(off_totals)
            print(f"🧮 Average Offloaded Tokens (total): {avg_off_total:.1f} tokens (avg small%: {avg_small_percentage:.2f}%)")
        if ans_off:
            avg_ans_off = sum(ans_off) / len(ans_off)
            avg_ans_ratio = (sum([r for r in ans_ratios if r is not None]) / len([r for r in ans_ratios if r is not None])) if ans_ratios else None
            ratio_str = f"{avg_ans_ratio*100:.2f}%" if avg_ans_ratio is not None else "N/A"
            print(f"🧮 Average Offloaded Tokens (answer_str): {avg_ans_off:.1f} tokens (offload ratio: {ratio_str})")
        if reas_off:
            avg_reas_off = sum(reas_off) / len(reas_off)
            avg_reas_ratio = (sum([r for r in reas_ratios if r is not None]) / len([r for r in reas_ratios if r is not None])) if reas_ratios else None
            ratio_str = f"{avg_reas_ratio*100:.2f}%" if avg_reas_ratio is not None else "N/A"
            print(f"🧮 Average Offloaded Tokens (reasoning_str): {avg_reas_off:.1f} tokens (offload ratio: {ratio_str})")


def main():
    """Main function to run RelayGen scoring."""
    parser = argparse.ArgumentParser(
        description="Evaluate RelayGen model outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        choices=["aime", "aime25", "math", "amc23", "gpqa"], 
        default="aime25",
        help="Dataset to evaluate"
    )
    
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="outputs/relaygen-qwen3_1.7b-qwen3_32b",
        help="Root directory containing results"
    )
    
    parser.add_argument(
        "--target_num", 
        type=int, 
        default=500,
        help="Number of problems to evaluate"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show per-problem statistics (length and accuracy)"
    )

    parser.add_argument(
        "--repeat_id",
        type=str,
        default=None,
        help="Filter by specific repeat ID (optional, e.g. '0' or '0,1,2')"
    )
    
    args, _ = parser.parse_known_args()
    
    # Parse repeat_id
    repeat_ids = None
    if args.repeat_id is not None:
        try:
            repeat_ids = [int(x.strip()) for x in args.repeat_id.split(',')]
        except ValueError:
            print(f"❌ Error: Invalid format for --repeat_id: '{args.repeat_id}'. Must be integer or comma-separated integers (e.g. '0,1,2').")
            return

    # Create scorer and run evaluation
    scorer = RelayGenScorer(
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        target_num=args.target_num
    )
    
    metadata_dict = scorer.score_relaygen(verbose=args.verbose, repeat_id=repeat_ids)
    
    print(f"\n✅ Evaluation completed for {len(metadata_dict)} problems.")


if __name__ == "__main__":
    main()
