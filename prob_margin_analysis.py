import os
import json
import gc
import argparse
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from utils_profile import VLLMOfflineProbMarginCalculator
from utils.utils import load_reasoning_strings_with_repeats_from_results
from utils.utils import DEFAULT_CUE_TOKENS, QWEN3_SWITCH_CUES

def _compute_offload_positions(tokens_local: Optional[List[str]], cue_tokens_local: Optional[List[str]]):
    if not tokens_local:
        return []
    cts = cue_tokens_local or DEFAULT_CUE_TOKENS
    end_markers = [".", "!", "?", "\n"]
    regions = []
    n = len(tokens_local)
    for i, tok in enumerate(tokens_local):
        ts = tok if isinstance(tok, str) else str(tok)
        t_clean = ts.strip()
        mc = None
        start = None

        next_clean = None
        if (i + 1) < n:
            next_tok = tokens_local[i + 1]
            next_clean = (next_tok if isinstance(next_tok, str) else str(next_tok)).strip()

        if mc is None:
            for cue in cts:
                if cue.endswith(","):
                    base = cue[:-1]
                    if (t_clean == base) or ts.startswith(" " + base):
                        if next_clean == ",":
                            mc = cue
                            start = i + 2
                            break
                elif cue.endswith(" "):
                    base = cue.rstrip()
                    if (t_clean == base) or ts.startswith(" " + base):
                        mc = cue
                        start = i + 1
                        break

        if mc is None:
            for cue in cts:
                c = cue
                if not c.strip():
                    continue
                if (c in t_clean) or (t_clean == c) or (ts == c):
                    mc = c
                    start = i + 1
                    break

        if mc is not None:
            start = start if start is not None else i + 1
            end = start
            while end < n:
                t = tokens_local[end]
                tt = t.strip() if isinstance(t, str) else str(t)
                if any(em in tt for em in end_markers):
                    break
                end += 1
            regions.append((start, end))
    positions = []
    for s, e in regions:
        positions.extend(list(range(s, min(e, n))))
    positions = sorted(set(positions))
    return positions


def _ensure_dir(d: str):
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def analyze_prob_margin_result_dir(
    model_path: str,
    result_dir: str,
    dataset_name: str,
    num_examples: int,
    save_dir: str,
    top_logprobs: int = 20,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 8192,
    chat_template_path: Optional[str] = None,
    cuda_devices: Optional[str] = None,
    use_qwen3_cues: bool = False,
    do_plot: bool = False,
    block_size: Optional[int] = 4,
    max_xtick_labels: int = 100,
    concat_answer: bool = False,
) -> dict:
    reasoning_strs, dataset, problem_indices, repeat_indices = load_reasoning_strings_with_repeats_from_results(
        result_dir=result_dir, num_examples=num_examples, dataset_name=dataset_name, concat_answer=concat_answer
    )

    if not reasoning_strs:
        return {"error": f"No reasoning strings found in {result_dir}"}

    _ensure_dir(save_dir)
    per_case_csv_path = os.path.join(save_dir, "prob_margin_stats.csv")
    if not os.path.exists(per_case_csv_path):
        with open(per_case_csv_path, "w", encoding="utf-8") as f:
            f.write(
                "dataset,problem_id,repeat_id,sequence_id,token_count,offload_token_count,offload_token_ratio,"
                "mean_margin_overall,mean_margin_offload,mean_margin_non_offload,mean_top1_logprob,mean_top2_logprob\n"
            )

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    calc = VLLMOfflineProbMarginCalculator(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        chat_template_path=chat_template_path,
    )

    outputs = []
    per_problem_acc = {}
    margins_means = []

    for i, text in enumerate(tqdm(reasoning_strs, desc="Prompt logprobs for prob margin")):
        pid = problem_indices[i] if (problem_indices and i < len(problem_indices)) else None
        rid = repeat_indices[i] if (repeat_indices and i < len(repeat_indices)) else None
        prompt_text = calc.build_text_to_analyze(text, dataset, dataset_name, pid)
        prompt_lps, toks = calc.get_prompt_logprobs_raw(
            text, top_logprobs=top_logprobs, dataset=dataset, problem_idx=pid, dataset_name=dataset_name, return_tokens=True
        )
        top1_list: List[float] = []
        top2_list: List[float] = []
        margin_list: List[float] = []
        for lp in prompt_lps:
            vals = [float(getattr(v, "logprob", 0.0)) for v in (lp or {}).values()]
            if len(vals) == 0:
                top1_list.append(0.0)
                top2_list.append(0.0)
                margin_list.append(0.0)
            elif len(vals) == 1:
                m1 = float(np.max(vals))
                top1_list.append(m1)
                top2_list.append(0.0)
                # Probability margin: exp(top1) - exp(top2) where top2 is -inf (prob 0)
                # exp(m1) - 0
                margin_list.append(float(np.exp(m1)))
            else:
                s = sorted(vals, reverse=True)
                top1_list.append(float(s[0]))
                top2_list.append(float(s[1]))
                # Probability margin
                p1 = np.exp(s[0])
                p2 = np.exp(s[1])
                margin_list.append(float(p1 - p2))

        cue_set = QWEN3_SWITCH_CUES if use_qwen3_cues else DEFAULT_CUE_TOKENS
        off_pos = _compute_offload_positions(toks or [], cue_set)
        off_margins = [margin_list[idx] for idx in off_pos if idx < len(margin_list)]
        non_margins = [margin_list[idx] for idx in range(len(margin_list)) if idx not in set(off_pos)]
        mean_margin_overall = float(np.mean(margin_list)) if margin_list else 0.0
        mean_margin_off = float(np.mean(off_margins)) if off_margins else 0.0
        mean_margin_non = float(np.mean(non_margins)) if non_margins else 0.0
        mean_top1 = float(np.mean(top1_list)) if top1_list else 0.0
        mean_top2 = float(np.mean(top2_list)) if top2_list else 0.0
        off_count = len(off_pos)
        total_count = len(margin_list)
        off_ratio = (float(off_count) / float(total_count)) if total_count > 0 else 0.0

        if rid is not None:
            seq_id = f"{dataset_name}_problem_{pid if pid is not None else i}_repeat_{rid}"
        else:
            seq_id = f"{dataset_name}_problem_{pid if pid is not None else i}"

        base = seq_id.replace("/", "_").replace(" ", "_")
        with open(os.path.join(save_dir, f"{base}_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt_text)
        json_path = os.path.join(save_dir, f"{base}_margin.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sequence_id": seq_id,
                    "positions": list(range(len(margin_list))),
                    "tokens": toks or [],
                    "top1_logprob": top1_list,
                    "top2_logprob": top2_list,
                    "prob_margin": margin_list,
                    "mean_margin_overall": mean_margin_overall,
                    "mean_margin_offload": mean_margin_off,
                    "mean_margin_non_offload": mean_margin_non,
                    "mean_top1_logprob": mean_top1,
                    "mean_top2_logprob": mean_top2,
                    "offload_token_ratio": off_ratio,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(per_case_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{dataset_name},{pid if pid is not None else ''},{rid if rid is not None else ''},{seq_id},{total_count},{off_count},{off_ratio},{mean_margin_overall},{mean_margin_off},{mean_margin_non},{mean_top1},{mean_top2}\n"
            )

        png_path = None
        if do_plot:
            cue_set = QWEN3_SWITCH_CUES if use_qwen3_cues else None
            png_path = plot_margin_series(
                save_dir,
                seq_id,
                margin_list,
                tokens=toks or [],
                cue_tokens=cue_set,
                max_xtick_labels=max_xtick_labels,
                block_size=block_size,
            )

        outputs.append({"sequence_id": seq_id, "json": json_path, "png": png_path})
        margins_means.append(mean_margin_overall)

        pid_key = pid if pid is not None else i
        if pid_key not in per_problem_acc:
            per_problem_acc[pid_key] = []
        per_problem_acc[pid_key].append({
            "total_count": total_count,
            "off_count": off_count,
            "off_ratio": off_ratio,
            "mean_margin_overall": mean_margin_overall,
            "mean_margin_off": mean_margin_off,
            "mean_margin_non": mean_margin_non,
            "mean_top1": mean_top1,
            "mean_top2": mean_top2,
        })

    try:
        del calc
        gc.collect()
    except Exception:
        pass

    per_problem_csv_path = os.path.join(save_dir, "prob_margin_stats_by_problem.csv")
    with open(per_problem_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "dataset,problem_id,sequence_count,mean_token_count,mean_offload_token_count,mean_offload_token_ratio,"
            "mean_margin_overall,mean_margin_offload,mean_margin_non_offload,mean_top1_logprob,mean_top2_logprob\n"
        )
        for pid_key in sorted(per_problem_acc.keys()):
            rows = per_problem_acc[pid_key]
            mean_total = float(np.mean([r["total_count"] for r in rows])) if rows else 0.0
            mean_off = float(np.mean([r["off_count"] for r in rows])) if rows else 0.0
            mean_off_ratio = float(np.mean([r["off_ratio"] for r in rows])) if rows else 0.0
            mean_margin_overall = float(np.mean([r["mean_margin_overall"] for r in rows])) if rows else 0.0
            mean_margin_off = float(np.mean([r["mean_margin_off"] for r in rows])) if rows else 0.0
            mean_margin_non = float(np.mean([r["mean_margin_non"] for r in rows])) if rows else 0.0
            mean_top1 = float(np.mean([r["mean_top1"] for r in rows])) if rows else 0.0
            mean_top2 = float(np.mean([r["mean_top2"] for r in rows])) if rows else 0.0
            f.write(
                f"{dataset_name},{pid_key},{len(rows)},{mean_total},{mean_off},{mean_off_ratio},{mean_margin_overall},{mean_margin_off},{mean_margin_non},{mean_top1},{mean_top2}\n"
            )

    summary_path = os.path.join(save_dir, "prob_margin_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_sequences": len(outputs),
                "mean_margin_overall_avg": float(np.mean(margins_means)) if margins_means else 0.0,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "total_sequences": len(outputs),
        "per_sequence_outputs": outputs,
        "summary_json": summary_path,
        "per_case_csv": per_case_csv_path,
        "per_problem_csv": per_problem_csv_path,
    }


def analyze_cue_margin_effects(
    model_path: str,
    result_dir: str,
    dataset_name: str,
    num_examples: int,
    save_dir: str,
    candidate_cues: Optional[List[str]] = None,
    top_logprobs: int = 20,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 8192,
    chat_template_path: Optional[str] = None,
    cuda_devices: Optional[str] = None,
    min_occurrences: int = 1,
    min_delta: float = 0.0,
    concat_answer: bool = False,
) -> dict:
    reasoning_strs, dataset, problem_indices, repeat_indices = load_reasoning_strings_with_repeats_from_results(
        result_dir=result_dir, num_examples=num_examples, dataset_name=dataset_name, concat_answer=concat_answer
    )

    if not reasoning_strs:
        return {"error": f"No reasoning strings found in {result_dir}"}
    _ensure_dir(save_dir)
    cues = list(candidate_cues or DEFAULT_CUE_TOKENS)
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    calc = VLLMOfflineProbMarginCalculator(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        chat_template_path=chat_template_path,
    )
    cue_stats = {c: {"occ": 0, "off_margins": [], "non_margins": []} for c in cues}
    for i, text in enumerate(tqdm(reasoning_strs, desc="Cue margin effects")):
        pid = problem_indices[i] if (problem_indices and i < len(problem_indices)) else None
        rid = repeat_indices[i] if (repeat_indices and i < len(repeat_indices)) else None
        
        prompt_lps, toks = calc.get_prompt_logprobs_raw(
            text, top_logprobs=top_logprobs, dataset=dataset, problem_idx=pid, dataset_name=dataset_name, return_tokens=True
        )

        vals = []
        top1_list = []
        top2_list = []
        
        for lp in prompt_lps:
            v = [float(getattr(x, "logprob", 0.0)) for x in (lp or {}).values()]
            if len(v) == 0:
                vals.append(0.0)
                top1_list.append(0.0)
                top2_list.append(0.0)
            elif len(v) == 1:
                # Probability margin
                m1 = float(np.max(v))
                top1_list.append(m1)
                top2_list.append(0.0)
                vals.append(float(np.exp(m1)))
            else:
                s = sorted(v, reverse=True)
                top1_list.append(float(s[0]))
                top2_list.append(float(s[1]))
                # Probability margin
                p1 = np.exp(s[0])
                p2 = np.exp(s[1])
                vals.append(float(p1 - p2))
                
        # Save per-sequence margin JSON
        if rid is not None:
            seq_id = f"{dataset_name}_problem_{pid if pid is not None else i}_repeat_{rid}"
        else:
            seq_id = f"{dataset_name}_problem_{pid if pid is not None else i}"
            
        base = seq_id.replace("/", "_").replace(" ", "_")
        json_path = os.path.join(save_dir, f"{base}_margin.json")
        
        # Calculate means for the JSON
        mean_margin_overall = float(np.mean(vals)) if vals else 0.0
        
        # We need offload positions for the JSON stats too if we want to be consistent, 
        # but analyze_cue_margin_effects iterates over ALL cues to aggregate stats.
        # For the per-sequence JSON, we can compute offload stats based on ALL candidate cues (union).
        off_pos_all = _compute_offload_positions(toks or [], cues)
        off_margins_all = [vals[idx] for idx in off_pos_all if idx < len(vals)]
        non_margins_all = [vals[idx] for idx in range(len(vals)) if idx not in set(off_pos_all)]
        
        mean_margin_off = float(np.mean(off_margins_all)) if off_margins_all else 0.0
        mean_margin_non = float(np.mean(non_margins_all)) if non_margins_all else 0.0
        mean_top1 = float(np.mean(top1_list)) if top1_list else 0.0
        mean_top2 = float(np.mean(top2_list)) if top2_list else 0.0
        off_count = len(off_pos_all)
        total_count = len(vals)
        off_ratio = (float(off_count) / float(total_count)) if total_count > 0 else 0.0
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sequence_id": seq_id,
                    "positions": list(range(len(vals))),
                    "tokens": toks or [],
                    "top1_logprob": top1_list,
                    "top2_logprob": top2_list,
                    "prob_margin": vals,
                    "mean_margin_overall": mean_margin_overall,
                    "mean_margin_offload": mean_margin_off,
                    "mean_margin_non_offload": mean_margin_non,
                    "mean_top1_logprob": mean_top1,
                    "mean_top2_logprob": mean_top2,
                    "offload_token_ratio": off_ratio,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        for cue in cues:
            off_pos = _compute_offload_positions(toks or [], [cue])
            if off_pos:
                cue_stats[cue]["occ"] += 1
                off_ms = [vals[idx] for idx in off_pos if idx < len(vals)]
                non_ms = [vals[idx] for idx in range(len(vals)) if idx not in set(off_pos)]
                if off_ms:
                    cue_stats[cue]["off_margins"].extend(off_ms)
                if non_ms:
                    cue_stats[cue]["non_margins"].extend(non_ms)
    try:
        del calc
        gc.collect()
    except Exception:
        pass
    rows = []
    for cue in cues:
        occ = cue_stats[cue]["occ"]
        
        # Calculate stats for offload margins
        if cue_stats[cue]["off_margins"]:
            off_arr = np.array(cue_stats[cue]["off_margins"])
            off_mean = float(np.mean(off_arr))
            off_std = float(np.std(off_arr))
            off_sem = float(off_std / np.sqrt(len(off_arr)))
        else:
            off_mean, off_std, off_sem = 0.0, 0.0, 0.0
            
        # Calculate stats for non-offload margins
        if cue_stats[cue]["non_margins"]:
            non_arr = np.array(cue_stats[cue]["non_margins"])
            non_mean = float(np.mean(non_arr))
            non_std = float(np.std(non_arr))
            non_sem = float(non_std / np.sqrt(len(non_arr)))
        else:
            non_mean, non_std, non_sem = 0.0, 0.0, 0.0
            
        delta = off_mean - non_mean
        
        rows.append({
            "cue": cue, 
            "occurrences": occ, 
            "mean_offload_margin": off_mean,
            "std_offload_margin": off_std,
            "sem_offload_margin": off_sem,
            "mean_non_offload_margin": non_mean,
            "std_non_offload_margin": non_std,
            "sem_non_offload_margin": non_sem,
            "delta": delta
        })
        
    rows_sorted = sorted(rows, key=lambda r: r["delta"], reverse=True)
    filtered = [r for r in rows_sorted if (r["occurrences"] >= min_occurrences and r["delta"] > min_delta)]
    csv_path = os.path.join(save_dir, "cue_margin_effects.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("cue,occurrences,mean_offload_margin,std_offload_margin,sem_offload_margin,mean_non_offload_margin,std_non_offload_margin,sem_non_offload_margin,delta\n")
        for r in rows_sorted:
            f.write(f"{r['cue']},{r['occurrences']},{r['mean_offload_margin']},{r['std_offload_margin']},{r['sem_offload_margin']},{r['mean_non_offload_margin']},{r['std_non_offload_margin']},{r['sem_non_offload_margin']},{r['delta']}\n")
    json_path = os.path.join(save_dir, "cue_margin_effects.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sorted": rows_sorted, "filtered": filtered}, f, ensure_ascii=False, indent=2)
    return {"csv": csv_path, "json": json_path, "sorted": rows_sorted, "filtered": filtered}


def main():
    parser = argparse.ArgumentParser(description="Top1-Top2 prob margin analyzer")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="aime25")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="margin_outputs")
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--chat_template_path", type=str, default=None)
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--use_qwen3_cues", action="store_true")
    parser.add_argument("--cue_effects", action="store_true")
    parser.add_argument("--min_occurrences", type=int, default=1)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--max_xtick_labels", type=int, default=100)
    parser.add_argument("--concat_answer", action="store_true")
    args = parser.parse_args()

    _ensure_dir(args.save_dir)
    if args.cue_effects:
        res = analyze_cue_margin_effects(
            model_path=args.model_path,
            result_dir=args.result_dir,
            dataset_name=args.dataset_name,
            num_examples=args.num_examples,
            save_dir=args.save_dir,
            candidate_cues=DEFAULT_CUE_TOKENS,
            top_logprobs=args.top_logprobs,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            chat_template_path=args.chat_template_path,
            cuda_devices=args.cuda,
            min_occurrences=args.min_occurrences,
            min_delta=args.min_delta,
            concat_answer=args.concat_answer,
        )
    else:
        res = analyze_prob_margin_result_dir(
            model_path=args.model_path,
            result_dir=args.result_dir,
            dataset_name=args.dataset_name,
            num_examples=args.num_examples,
            save_dir=args.save_dir,
            top_logprobs=args.top_logprobs,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            chat_template_path=args.chat_template_path,
            cuda_devices=args.cuda,
            use_qwen3_cues=args.use_qwen3_cues,
            do_plot=bool(args.plot and _MATPLOTLIB_AVAILABLE),
            block_size=args.block_size,
            max_xtick_labels=args.max_xtick_labels,
            concat_answer=args.concat_answer,
        )
    print(json.dumps(res, ensure_ascii=False, indent=2))


try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False

def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)[:200]

def plot_margin_series(
    save_dir: str,
    seq_id: str,
    margins: List[float],
    tokens: Optional[List[str]] = None,
    cue_tokens: Optional[List[str]] = None,
    max_xtick_labels: int = 100,
    block_size: Optional[int] = 4,
) -> Optional[str]:
    if not _MATPLOTLIB_AVAILABLE:
        return None
    _ensure_dir(save_dir)
    base = _safe_filename(seq_id)
    png_path = os.path.join(save_dir, f"{base}_margin.png")
    n_total = len(margins)
    if block_size is None or block_size <= 1:
        series = margins
        x_vals = list(range(1, n_total + 1))
    else:
        k = max(1, int(block_size))
        series = [float(np.mean(margins[i:i+k])) for i in range(0, n_total, k)]
        x_vals = [min(i + k, n_total) for i in range(0, n_total, k)]
    plt.figure(figsize=(12, 4))
    plt.plot(x_vals, series, color="#2A7FFF", linewidth=1.5)
    if cue_tokens:
        off_pos = _compute_offload_positions(tokens or [], cue_tokens)
        if off_pos:
            for pos in off_pos:
                plt.axvspan(pos, pos+1, color="#FFCC66", alpha=0.2)
    step = max(1, int(len(x_vals) / max(1, max_xtick_labels)))
    tick_positions = x_vals[::step]
    plt.xticks(tick_positions)
    plt.xlabel("Token count")
    plt.title(seq_id)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    return png_path
if __name__ == "__main__":
    main()
