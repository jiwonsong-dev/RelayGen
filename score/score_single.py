import os
import ast
import argparse
import pickle
from typing import Dict, List, Any, Optional, Tuple

from utils_score.parser import extract_answer
from utils_score.grader import math_equal
from utils.utils import get_dataset

# from transformers import AutoTokenizer

class SingleModelScorer:
    """Scorer for evaluating single model outputs."""
    
    def __init__(self, root_dir: str = "outputs/32b", dataset_name: str = "aime", target_num: int = 500):
        """
        Initialize the SingleModelScorer.
        
        Args:
            root_dir: Root directory containing model outputs
            dataset_name: Name of the dataset to evaluate
            target_num: Number of problems to evaluate
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.target_num = target_num
        self.root_path = None
        self.dataset = None  # Cache dataset to avoid repeated loading
        
    def _find_matching_folder(self) -> Optional[str]:
        """Find the folder that contains the dataset."""
        available_folders = os.listdir(self.root_dir)
        matching_folder = None
        
        # Order matters: more specific names first to avoid aime matching aime25
        known_datasets = ["aime", "aime25", "math", "amc23", "gpqa"]
        
        # First, try to find exact or best match for the specified dataset_name
        for folder in available_folders:
            if self.dataset_name in folder:
                matching_folder = folder
                break
                # Check if we need to update to a more specific dataset name
                # for dataset in known_datasets:
                #     if dataset in folder and dataset != self.dataset_name:
                #         # Only update if the found dataset is more specific
                #         # if len(dataset) > len(self.dataset_name) and self.dataset_name in dataset:
                #         #     self.dataset_name = dataset
                #         #     print(f"Updated dataset_name to more specific: {dataset} based on folder: {folder}")
                #         # break
                # break
        
        if matching_folder is None:
            # Try to find any folder that contains a known dataset name
            for folder in available_folders:
                for dataset in known_datasets:
                    if dataset in folder:
                        matching_folder = folder
                        # Auto-detect dataset_name from folder name
                        self.dataset_name = dataset
                        print(f"Auto-detected dataset: {dataset} from folder: {folder}")
                        break
                if matching_folder:
                    break
        
        return matching_folder
    
    def _setup_paths(self) -> None:
        """Setup the root path for evaluation."""
        matching_folder = self._find_matching_folder()
        
        if matching_folder is None:
            # Fallback to original behavior
            self.root_path = os.path.join(self.root_dir, self.dataset_name)
        else:
            self.root_path = os.path.join(self.root_dir, matching_folder)
    
    def _extract_ground_truth_and_prediction(self, data: Dict[str, Any], problem_id: int, output: str) -> Tuple[str, str]:
        """Extract ground truth and prediction answers based on dataset type."""
        # Load dataset only once and cache it
        if self.dataset is None:
            self.dataset = get_dataset(self.dataset_name)
        dataset = self.dataset
        
        if self.dataset_name == 'aime':
            gt_answer = dataset['answer'][problem_id-60]
            pred_answer = extract_answer(output, 'aime24', use_last_number=False)
        elif self.dataset_name == 'aime25':
            gt_answer = str(dataset['answer'][problem_id])
            pred_answer = extract_answer(output, 'aime25', use_last_number=False)
        elif self.dataset_name == 'math':
            gt_answer = extract_answer(dataset['solution'][problem_id], 'math-oai', use_last_number=False)
            pred_answer = extract_answer(output, 'math-oai', use_last_number=False)
        elif self.dataset_name == "amc23":
            gt_answer = str(dataset['answer'][problem_id])
            pred_answer = extract_answer(output, 'amc23', use_last_number=False)
        elif self.dataset_name == "gpqa":
            gt_answer = data[-1]['correct_answer']
            pred_answer = extract_answer(output, 'gpqa', use_last_number=False)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        return gt_answer, pred_answer
    
    def evaluate(self, verbose: bool = True) -> Dict[int, Dict[str, List]]:
        """Evaluate single model outputs and return results."""
        self._setup_paths()
        
        total_responses = 0
        total_len = 0
        got_rights = 0
        
        # Switching statistics tracking
        switching_data = []
        has_switching_stats = False
        
        analysis_targets = sorted(os.listdir(self.root_path), key=int)[:self.target_num]
        metadata_dict = {}
        
        for subfolder_name in analysis_targets:
            subfolder_path = os.path.join(self.root_path, subfolder_name)
            
            if os.path.isdir(subfolder_path):
                problem_id = int(subfolder_path.split('/')[-1])
                
                metadata_dict[problem_id] = {
                    'correctness': [],
                    'gen_len': []
                }
                
                for filename in sorted(os.listdir(subfolder_path), key=lambda x: int(os.path.splitext(x)[0])):
                    if filename.endswith(".pickle"):
                        file_path = os.path.join(subfolder_path, filename)
                        
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        total_responses += 1
                        
                        # Support both 'total_tokens' and 'num_output_tokens' keys
                        if 'total_tokens' in data[-1]:
                            gen_len = data[-1]['total_tokens']
                        elif 'num_output_tokens' in data[-1]:
                            gen_len = data[-1]['num_output_tokens']
                        else:
                            gen_len = 0  # fallback
                            
                        total_len += gen_len
                        metadata_dict[problem_id]['gen_len'].append(gen_len)
                        
                        # Collect switching statistics if available
                        if ('switching_stats' in data[-1] and 
                            data[-1]['switching_stats'] is not None and 
                            data[-1]['switching_stats'] != {} and 
                            'large_model_percentage' in data[-1]['switching_stats']):
                            has_switching_stats = True
                            switching_data.append(data[-1]['switching_stats'])
                        
                        # Support both 'final_text' and 'answer_str' keys
                        if 'final_text' in data[-1]:
                            output = data[-1]['final_text']
                        elif 'answer_str' in data[-1]:
                            output = data[-1]['answer_str']
                        else:
                            output = None  # fallback
                        
                        if output is None:  # budget overflow
                            right = False
                        else:
                            gt_answer, pred_answer = self._extract_ground_truth_and_prediction(data, problem_id, output)
                            right = math_equal(pred_answer, gt_answer, timeout=False)
                        
                        if right:
                            got_rights += 1
                            metadata_dict[problem_id]['correctness'].append(1)
                        else:
                            metadata_dict[problem_id]['correctness'].append(0)
        
        # Always display results, but control per-problem details with verbose
        self._display_results(total_responses, total_len, got_rights, has_switching_stats, switching_data, metadata_dict, verbose)
        
        return metadata_dict
    
    def _display_results(self, total_responses: int, total_len: int, got_rights: int, 
                        has_switching_stats: bool, switching_data: List[Dict], metadata_dict: Dict, verbose: bool = True) -> None:
        """Display evaluation results in a beautiful format."""
        print("\n" + "=" * 60)
        print("🎯 SINGLE MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"📁 Root directory: {self.root_path}")
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
    
    def _print_switching_stats(self, switching_data: List[Dict]) -> None:
        """Print switching statistics for single model (if available)."""
        print("\n" + "-" * 60)
        print("🔄 SWITCHING STATISTICS")
        print("-" * 60)
        
        avg_large_percentage = sum(stats['large_model_percentage'] for stats in switching_data) / len(switching_data)
        avg_small_percentage = sum(stats['small_model_percentage'] for stats in switching_data) / len(switching_data)
        avg_switches = sum(stats['total_switches'] for stats in switching_data) / len(switching_data)
        avg_switch_rate = sum(stats['switch_rate'] for stats in switching_data) / len(switching_data)
        
        print(f"🔵 Average Base Model Generation: {avg_large_percentage:.2f}%")
        print(f"🟢 Average Small Model Generation: {avg_small_percentage:.2f}%")
        print(f"🔄 Average Total Switches: {avg_switches:.1f}")
        print(f"📊 Average Switch Rate: {avg_switch_rate:.4f}")


def score_single(args, verbose=True):
    """Legacy function for backward compatibility."""
    scorer = SingleModelScorer(
        root_dir=args.root_dir,
        dataset_name=args.dataset_name,
        target_num=args.target_num
    )
    return scorer.evaluate(verbose=verbose)


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Analyze single model outputs")
    parser.add_argument("--dataset_name", type=str, choices=["aime", "aime25", "math", "amc23", "gpqa"], default="aime",
                        help="Dataset to evaluate")
    parser.add_argument("--root_dir", type=str, default="outputs/32b",
                        help="Root directory containing model outputs")
    parser.add_argument("--target_num", type=int, default=500,
                        help="Number of problems to evaluate")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-problem statistics (length and accuracy)")
    args, _ = parser.parse_known_args()

    # Create scorer and run evaluation
    scorer = SingleModelScorer(
        root_dir=args.root_dir,
        dataset_name=args.dataset_name,
        target_num=args.target_num
    )
    
    results = scorer.evaluate(verbose=args.verbose)
    return results


if __name__ == "__main__":
    main()