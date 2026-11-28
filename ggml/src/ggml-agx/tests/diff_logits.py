#!/usr/bin/env python3
"""Differential logit parity & 0-NaN/Inf verifier for ggml-agx vs reference backend.

Compares raw F32 logit dumps (or `llama-logits` / `llama-perplexity` / custom binary output)
between AGX0 and CPU/Vulkan0 reference on Inkling 2 Mini and DSpark.

Usage:
    diff_logits.py --ref ref_logits.bin --cand agx_logits.bin [--vocab 200058] [--tol 5e-2] [--min-cos 0.999]
    diff_logits.py --check-text-dump ref.txt cand.txt
"""
import argparse
import math
import struct
import sys
from pathlib import Path


def load_f32_bin(path: Path):
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path}: size {len(data)} not a multiple of 4 bytes")
    n = len(data) // 4
    return struct.unpack(f"<{n}f", data)


def topk(vals, k=5):
    indexed = sorted(enumerate(vals), key=lambda kv: kv[1], reverse=True)
    return indexed[:k]


def compare_vectors(ref, cand, tol=5e-2, min_cos=0.999):
    if len(ref) != len(cand):
        raise ValueError(f"Length mismatch: ref={len(ref)} vs cand={len(cand)}")
    n = len(ref)
    nan_ref = sum(1 for v in ref if math.isnan(v))
    inf_ref = sum(1 for v in ref if math.isinf(v))
    nan_cand = sum(1 for v in cand if math.isnan(v))
    inf_cand = sum(1 for v in cand if math.isinf(v))

    if nan_cand > 0 or inf_cand > 0:
        return {
            "pass": False,
            "reason": f"AGX0 produced non-finite logits: NaN={nan_cand}, Inf={inf_cand}",
            "n": n,
            "nan_cand": nan_cand,
            "inf_cand": inf_cand,
        }

    max_abs = 0.0
    max_idx = -1
    sum_sq = 0.0
    dot = 0.0
    nr = 0.0
    nc = 0.0
    for i, (r, c) in enumerate(zip(ref, cand)):
        d = abs(r - c)
        if d > max_abs:
            max_abs = d
            max_idx = i
        sum_sq += d * d
        dot += r * c
        nr += r * r
        nc += c * c

    rmse = math.sqrt(sum_sq / max(n, 1))
    cos = dot / (math.sqrt(nr) * math.sqrt(nc)) if (nr > 0.0 and nc > 0.0) else 0.0

    ref_top5 = topk(ref, 5)
    cand_top5 = topk(cand, 5)
    top1_match = ref_top5[0][0] == cand_top5[0][0]
    top5_overlap = len({i for i, _ in ref_top5} & {i for i, _ in cand_top5})

    ok = (nan_cand == 0 and inf_cand == 0 and cos >= min_cos and max_abs <= tol and top1_match)
    return {
        "pass": ok,
        "n": n,
        "nan_ref": nan_ref,
        "inf_ref": inf_ref,
        "nan_cand": nan_cand,
        "inf_cand": inf_cand,
        "max_abs": max_abs,
        "max_idx": max_idx,
        "rmse": rmse,
        "cosine": cos,
        "top1_match": top1_match,
        "ref_top1": ref_top5[0],
        "cand_top1": cand_top5[0],
        "top5_overlap": top5_overlap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="Reference F32 logits binary")
    ap.add_argument("--cand", required=True, type=Path, help="Candidate AGX0 F32 logits binary")
    ap.add_argument("--tol", type=float, default=5e-2, help="Max allowed abs logit error")
    ap.add_argument("--min-cos", type=float, default=0.999, help="Min allowed cosine similarity")
    args = ap.parse_args()

    ref = load_f32_bin(args.ref)
    cand = load_f32_bin(args.cand)
    res = compare_vectors(ref, cand, tol=args.tol, min_cos=args.min_cos)

    print(f"Differential Logit Parity ({res['n']} logits):")
    print(f"  NaN/Inf (cand): {res['nan_cand']}/{res['inf_cand']}")
    if "max_abs" in res:
        print(f"  max_abs_err:    {res['max_abs']:.6g} (at idx {res['max_idx']})")
        print(f"  rmse:           {res['rmse']:.6g}")
        print(f"  cosine_sim:     {res['cosine']:.8f}")
        print(f"  top1_match:     {res['top1_match']} (ref={res['ref_top1']}, cand={res['cand_top1']})")
        print(f"  top5_overlap:   {res['top5_overlap']}/5")
    if not res["pass"]:
        print(f"FAIL: {res.get('reason', 'threshold exceeded')}", file=sys.stderr)
        return 1
    print("PASS: 0-NaN/Inf differential logit parity verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
