#!/usr/bin/env python3
"""Compile-validate all ggml-agx GLSL 450 compute shaders with glslangValidator + spirv-val."""
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"

# Optional extra defines per shader so every specialized variant gets checked too.
EXTRA_VARIANTS = {
    "nvfp4_vecmat.comp": [
        ["-DREPACKED=1", "-DBATCH_M=1", "-DK_SGS=1"],
        ["-DREPACKED=1", "-DBATCH_M=5", "-DK_SGS=4"],
        ["-DREPACKED=0", "-DBATCH_M=1", "-DK_SGS=1"],
    ],
    "mxfp8_q8_vecmat.comp": [
        ["-DQUANT_MODE=0", "-DBATCH_M=1"],
        ["-DQUANT_MODE=1", "-DBATCH_M=1"],
        ["-DQUANT_MODE=2", "-DBATCH_M=1"],
        ["-DQUANT_MODE=4", "-DBATCH_M=1"],
        ["-DQUANT_MODE=6", "-DBATCH_M=1"],
    ],
    "agx_mma_gemm.comp": [
        ["-DTILE_MODE=0"],
        ["-DTILE_MODE=1"],
    ],
}


def main():
    os.chdir(KERNEL_DIR)
    shaders = sorted(glob.glob("*.comp"))
    if not shaders:
        print("No .comp shaders found in", KERNEL_DIR)
        return 1

    fails = 0
    with tempfile.TemporaryDirectory(prefix="agx-spv-") as td:
        for s in shaders:
            variants = EXTRA_VARIANTS.get(s, [[]])
            for defs in variants:
                out_spv = os.path.join(td, "out.spv")
                cmd = ["glslangValidator", "-V", "--target-env", "vulkan1.3", "--spirv-val", "-I."] + defs + [s, "-o", out_spv]
                r = subprocess.run(cmd, capture_output=True, text=True)
                label = f"{s} {' '.join(defs)}".strip()
                if r.returncode == 0:
                    print(f"[PASS] {label}")
                else:
                    fails += 1
                    print(f"[FAIL] {label} (rc={r.returncode})")
                    for line in (r.stdout + r.stderr).splitlines()[:15]:
                        print("   ", line)
    print(f"\nGLSL/SPIR-V validation complete: fails={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
