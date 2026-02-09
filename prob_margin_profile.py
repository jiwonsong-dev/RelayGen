import os
import json
import csv
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

from utils.utils import DEFAULT_CUE_TOKENS

def compute_cue_segments(tokens, margins, cues):
    end_markers = [".", "!", "?", "\n"]
    n = len(tokens)
    
    cue_margins = {c: [] for c in cues}
    cue_counts = {c: 0 for c in cues}
    
    overall_margins = margins
    
    for i, tok in enumerate(tokens):
        ts = tok if isinstance(tok, str) else str(tok)
        t_clean = ts.strip()
        matched_cue = None
        start = None

        # Generic multi-token handling for cues ending with ',' or ' '
        next_clean = None
        if (i + 1) < n:
            next_tok = tokens[i + 1]
            next_clean = (next_tok if isinstance(next_tok, str) else str(next_tok)).strip()
        if matched_cue is None:
            for cue in cues:
                if cue.endswith(","):
                    base = cue[:-1]
                    if (t_clean == base) or ts.startswith(" " + base):
                        if next_clean == ",":
                            matched_cue = cue
                            start = i + 2
                            break
                elif cue.endswith(" "):
                    base = cue.rstrip()
                    if (t_clean == base) or ts.startswith(" " + base):
                        matched_cue = cue
                        start = i + 1
                        break

        # General case-sensitive matching for all other cues
        if matched_cue is None:
            for cue in cues:
                c = cue
                if not c.strip():
                    continue
                if (c in t_clean) or (t_clean == c) or (ts == c):
                    matched_cue = cue
                    start = i + 1
                    break
        
        if matched_cue:
            start = (start if start is not None else i + 1)
            
            end = start
            while end < n:
                t = tokens[end]
                tt = t.strip() if isinstance(t, str) else str(t)
                if any(em in tt for em in end_markers):
                    break
                end += 1
            
            if start < end:
                segment_margins = margins[start:end]
                if segment_margins:
                    cue_margins[matched_cue].extend(segment_margins)
                    cue_counts[matched_cue] += 1

    return cue_margins, cue_counts

def main():
    parser = argparse.ArgumentParser(description="Profile margin stats per cue from JSON files")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing _margin.json files")
    parser.add_argument("--cues", type=str, default=None, help="Comma-separated list of cues. If not provided, uses ALL default cues.")
    parser.add_argument("--min_count", type=int, default=5, help="Minimum number of occurrences to report")
    parser.add_argument("--save_path", type=str, default="margin_profile_stats.csv", help="Path to save CSV stats")
    parser.add_argument("--delimiter", type=str, default=",", help="Delimiter to use for CSV output")
    parser.add_argument("--sample", type=int, default=None, help="Number of files to sample (must be a positive integer)")
    args = parser.parse_args()

    if args.cues:
        cues = [c.strip() for c in args.cues.split(',')]
    else:
        # Use ALL default cues if none provided
        cues = DEFAULT_CUE_TOKENS
    
    json_files = glob(os.path.join(args.json_dir, "*_margin.json"))
    if not json_files:
        print(f"No *_margin.json files found in {args.json_dir}")
        return

    # Sample files if requested
    if args.sample is not None and args.sample > 0:
        import random
        if len(json_files) > args.sample:
            print(f"Sampling {args.sample} files from {len(json_files)} total files.")
            json_files = random.sample(json_files, args.sample)
        else:
            print(f"Requested sample size {args.sample} is larger than total files {len(json_files)}. Using all files.")

    # Aggregate stats
    cue_tokens_pool = {c: [] for c in cues} # Pool of all margin values
    cue_segment_counts = {c: 0 for c in cues} # Count of segments (occurrences)
    cue_file_counts = {c: 0 for c in cues} # Count of files containing the cue
    
    overall_tokens_pool = []
    
    print(f"Processing {len(json_files)} files...")
    
    for jf in tqdm(json_files):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tokens = data.get("tokens", [])
            margins = data.get("prob_margin", [])
            
            if not tokens or not margins:
                continue
                
            if len(tokens) != len(margins):
                min_len = min(len(tokens), len(margins))
                tokens = tokens[:min_len]
                margins = margins[:min_len]
            
            # Overall pool
            overall_tokens_pool.extend(margins)
            
            # Cue segments
            seg_margins_dict, seg_counts_dict = compute_cue_segments(tokens, margins, cues)
            
            for c in cues:
                if seg_margins_dict[c]:
                    cue_tokens_pool[c].extend(seg_margins_dict[c])
                if seg_counts_dict[c]:
                    cue_segment_counts[c] += seg_counts_dict[c]
                    cue_file_counts[c] += 1
                
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    # Report
    print("\n" + "="*80)
    print(f"{'Cue':<15} | {'FileCount':<10} | {'SegCount':<8} | {'Mean':<8} | {'Std':<8} | {'SEM':<8}")
    print("-" * 80)
    
    results = []
    
    # Overall
    if overall_tokens_pool:
        n_tokens = len(overall_tokens_pool)
        m = np.mean(overall_tokens_pool)
        s = np.std(overall_tokens_pool)
        sem = s / np.sqrt(n_tokens)
        n_files = len(json_files)
        # For overall, SegCount is hard to define (all segments?), using n_files
        print(f"{'Overall':<15} | {n_files:<10} | {'-':<8} | {m:<8.2f} | {s:<8.2f} | {sem:<8.2f}")
        results.append(("Overall", m, s, sem, n_files, "-"))
    
    cue_results = []
    for c in cues:
        vals = cue_tokens_pool[c]
        seg_occ = cue_segment_counts[c]
        file_occ = cue_file_counts[c]
        
        if file_occ > 0 and vals: # Use file_occ to filter or seg_occ?
             # args.min_count default is 5. 
             # Let's filter by seg_occ >= min_count OR file_occ >= 1?
             # User provided min_count usually applies to occurrences.
             if seg_occ >= args.min_count:
                m = np.mean(vals)
                s = np.std(vals)
                n_tokens = len(vals)
                sem = s / np.sqrt(n_tokens)
                cue_results.append((c, m, s, sem, file_occ, seg_occ))
        else:
            pass

    # Sort cue_results by Mean (index 1) descending
    cue_results.sort(key=lambda x: x[1], reverse=True)
    
    for r in cue_results:
        # r: (Cue, Mean, Std, SEM, FileCount, SegCount)
        print(f"{r[0]:<15} | {r[4]:<10} | {r[5]:<8} | {r[1]:<8.2f} | {r[2]:<8.2f} | {r[3]:<8.2f}")
        results.append(r)

    print("="*80)
    
    # Save to file
    if args.save_path:
        try:
            if args.save_path.lower().endswith('.json'):
                json_results = []
                for r in results:
                    # r: (Cue, Mean, Std, SEM, FileCount, SegCount)
                    item = {
                        "Cue": r[0],
                        "FileCount": r[4],
                        "SegCount": r[5],
                        "Mean": float(f"{r[1]:.4f}") if isinstance(r[1], (int, float)) else r[1],
                        "Std": float(f"{r[2]:.4f}") if isinstance(r[2], (int, float)) else r[2],
                        "SEM": float(f"{r[3]:.4f}") if isinstance(r[3], (int, float)) else r[3]
                    }
                    json_results.append(item)
                with open(args.save_path, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, indent=4)
                print(f"\nStats saved to {args.save_path}")
            else:
                with open(args.save_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=args.delimiter)
                    writer.writerow(["Cue", "FileCount", "SegCount", "Mean", "Std", "SEM"])
                    for r in results:
                        # r: (Cue, Mean, Std, SEM, FileCount, SegCount)
                        writer.writerow([r[0], r[4], r[5], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"])
                print(f"\nStats saved to {args.save_path}")
        except Exception as e:
            print(f"\nError saving file: {e}")

    # Generate command string for plots_margin_transition.py
    # Format: Label:Value:Error
    # We use SEM for error bars usually, but user asked for "std profile".
    # Error bars in plots usually represent uncertainty of the mean (SEM) or data spread (Std).
    # Given the smallish values (0.25, 0.42) in previous examples compared to margins (10-17),
    # 0.25 looks like SEM (or maybe small Std?).
    # If Std is large (e.g. 5.0), then 0.25 is definitely SEM.
    # I will output both so user can choose.
    
    pairs_str_sem = ", ".join([f"{r[0]}:{r[1]:.2f}:{r[3]:.2f}" for r in results])
    pairs_str_std = ", ".join([f"{r[0]}:{r[1]:.2f}:{r[2]:.2f}" for r in results])
    
    print("\n[Command for plots_margin_transition.py (using SEM)]:")
    print(f'python plots_margin_transition.py --pairs "{pairs_str_sem}" --save_dir figs --out margin_profile_sem')
    
    print("\n[Command for plots_margin_transition.py (using Std)]:")
    print(f'python plots_margin_transition.py --pairs "{pairs_str_std}" --save_dir figs --out margin_profile_std')

if __name__ == "__main__":
    main()
