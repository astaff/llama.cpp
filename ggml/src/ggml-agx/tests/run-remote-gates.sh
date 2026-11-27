#!/usr/bin/env bash
# Runs ggml-agx M4 Max hardware verification gates sequentially on m4-nixos,
# strictly wrapping every GPU execution in guarded-run.py and waiting politely
# on ~/memo/results/benchmark.lock so concurrent S24 jobs are never disturbed.
set -uo pipefail

BUILD="/home/astaff/llama/llama-ggml-agx/build"
SHELL_NIX="/home/astaff/llama/shell.nix"
LOGDIR="/tmp/ggml-agx-gate-logs"
mkdir -p "${LOGDIR}"

# Resolve nix-shell PATH & vulkan-loader libdir once (zero GPU use)
GLSL_BIN="/nix/store/s1n6p73vv6h76wprpri7div9rjb4azfv-glslang-16.4.0-bin/bin"
SHADERC_BIN="/nix/store/zk5mjxy954zxvvg8m82gjcv8k0l5aava-shaderc-2026.1-bin/bin"
SPV_BIN="/nix/store/xr0zkcqlgxvxx01xrdy44mhiwswhyz9g-spirv-tools-1.4.357.0/bin"
VK_LOADER_LIB="/nix/store/308j9nmk6pm0g23g91ila08xg1pkx77l-vulkan-loader-1.4.357.0/lib"
export PATH="${GLSL_BIN}:${SHADERC_BIN}:${SPV_BIN}:${PATH}"
export LD_LIBRARY_PATH="${VK_LOADER_LIB}:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
export VK_DRIVER_FILES="/run/opengl-driver/share/vulkan/icd.d/asahi_icd.aarch64.json"
export GGML_AGX_KERNEL_DIR="/home/astaff/llama/llama-ggml-agx/ggml/src/ggml-agx/kernels"

run_guarded_step() {
    local name="$1"
    shift
    local logfile="${LOGDIR}/${name}.log"
    local rc=75
    while [[ ${rc} -eq 75 ]]; do
        flock -x /home/astaff/memo/results/benchmark.lock true || true
        python3 /home/astaff/llama/guarded-run.py --timeout 300 "ggml-agx-${name}" -- "$@" >"${logfile}" 2>&1
        rc=$?
        if [[ ${rc} -eq 75 ]]; then
            sleep 1
        fi
    done
    echo "=== GATE ${name}: rc=${rc} ==="
    cat "${logfile}"
    local resdir
    resdir=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('results',''))" "${logfile}" 2>/dev/null || true)
    if [[ -n "${resdir}" && -d "${resdir}" ]]; then
        [[ -s "${resdir}/stdout" ]] && echo "--- ${name} stdout ---" && tail -n 80 "${resdir}/stdout"
        [[ -s "${resdir}/stderr" ]] && echo "--- ${name} stderr ---" && tail -n 60 "${resdir}/stderr"
    fi
    return ${rc}
}

STEP="${1:-gates}"
case "${STEP}" in
    smoke)
        run_guarded_step "smoke-drm-cdm" /tmp/ggml-agx-probe/smoke_agx_drm_cdm
        ;;
    shaderdb)
        run_guarded_step "shaderdb-all" "${BUILD}/bin/test_agx_shaderdb_all"
        ;;
    roofline)
        run_guarded_step "roofline" "${BUILD}/bin/test_agx_roofline"
        ;;
    gemm-roofline)
        run_guarded_step "gemm-roofline" "${BUILD}/bin/test_agx_gemm_roofline"
        ;;
    backend-ops)
        run_guarded_step "backend-ops" "${BUILD}/bin/test-backend-ops" test -b AGX0
        ;;
    gates)
        rc_all=0
        if [[ -x /tmp/ggml-agx-probe/smoke_agx_drm_cdm ]]; then
            run_guarded_step "smoke-drm-cdm" /tmp/ggml-agx-probe/smoke_agx_drm_cdm || rc_all=1
        fi
        run_guarded_step "shaderdb-all" "${BUILD}/bin/test_agx_shaderdb_all" || rc_all=1
        run_guarded_step "roofline" "${BUILD}/bin/test_agx_roofline" || rc_all=1
        run_guarded_step "backend-ops" "${BUILD}/bin/test-backend-ops" test -b AGX0 || rc_all=1
        exit ${rc_all}
        ;;
    *)
        echo "Usage: $0 {smoke|shaderdb|roofline|backend-ops|gates}"
        exit 2
        ;;
esac
