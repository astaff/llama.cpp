#!/usr/bin/env bash
# Master verification harness for ggml-agx on M4 Max (m4-nixos).
# Enforces all isolation rules:
#   - Never touches m4-macos, SEP, boot policy, ESP, NVRAM, or llama-s24 / build-s23 / mesa-retest-20261125.
#   - Every GPU command on m4-nixos runs through guarded-run.py so ~/memo/results/benchmark.lock serializes GPU access.
set -euo pipefail

LOCAL_REPO="/home/astaff/m4-rig/var/pr-prep/llama-ggml-agx"
SSH_CONF="/home/astaff/m4-rig/config/ssh/m4.conf"
REMOTE_HOST="m4-nixos"
REMOTE_REPO="/home/astaff/llama/llama-ggml-agx"
REMOTE_BUILD="${REMOTE_REPO}/build"
TLC_JAR="/nix/store/ly1wfczr6563s936rbgib5szhd8hiw1k-tlaplus-1.7.4/share/java/tla2tools.jar"
INKLING_MODEL="/home/astaff/llama/models/Inkling-2-Mini-UD-Q4_K_XL.gguf"
DSPARK_MODEL="/home/astaff/llama/models/dspark-Inkling-2-Mini-Q8_0.gguf"

ssh_m4() {
    ssh -F "${SSH_CONF}" "${REMOTE_HOST}" "$@"
}

run_guarded() {
    local step="$1"
    shift
    local cmd="$*"
    local rc=75
    while [[ ${rc} -eq 75 ]]; do
        # Wait politely in kernel flock queue until concurrent GPU job releases benchmark.lock
        ssh_m4 "su - astaff -s /run/current-system/sw/bin/bash -c 'flock -x /home/astaff/memo/results/benchmark.lock true'" || true
        set +e
        ssh_m4 "su - astaff -s /run/current-system/sw/bin/bash -c \"python3 /home/astaff/llama/guarded-run.py ggml-agx-${step} -- ${cmd}\""
        rc=$?
        set -e
        if [[ ${rc} -eq 75 ]]; then
            echo "[guarded] benchmark.lock contended (rc=75), retrying..." >&2
        fi
    done
    return ${rc}
}

step_tlc() {
    local tla="${LOCAL_REPO}/ggml/src/ggml-agx/formal/AgxBarrier.tla"
    local cfg="${LOCAL_REPO}/ggml/src/ggml-agx/formal/AgxBarrier.cfg"
    if [[ ! -f "${tla}" || ! -f "${cfg}" ]]; then
        echo "[tlc] SKIP: ${tla} or ${cfg} not present yet"
        return 1
    fi
    echo "[tlc] Checking ${tla} with TLC..."
    local tmpdir
    tmpdir=$(mktemp -d /tmp/agx-tlc-XXXXXX)
    trap 'rm -rf "${tmpdir}"' RETURN
    java -XX:+UseParallelGC -cp "${TLC_JAR}" tlc2.TLC \
        -workers auto -metadir "${tmpdir}" -config "${cfg}" "${tla}"
}

step_local_numerics() {
    echo "[numerics] Compiling & running standalone test_agx_numerics locally..."
    local bin="/tmp/test_agx_numerics"
    g++ -O2 -std=c++17 -Wall -Wextra \
        "${LOCAL_REPO}/ggml/src/ggml-agx/tests/test_agx_numerics.cpp" -o "${bin}"
    "${bin}"
}

step_local_repack() {
    echo "[repack] Compiling & running test_agx_repack against agx-repack.cpp locally..."
    mkdir -p /tmp/ggml-inc
    printf '#define GGML_VERSION "1.1.0"\n#define GGML_VERSION_MAJOR 1\n#define GGML_VERSION_MINOR 1\n#define GGML_VERSION_PATCH 0\n#define GGML_COMMIT "0"\n' \
        > /tmp/ggml-inc/ggml-version.h
    gcc -O2 -ffunction-sections -I/tmp/ggml-inc -I"${LOCAL_REPO}/ggml/include" -I"${LOCAL_REPO}/ggml/src" \
        -c "${LOCAL_REPO}/ggml/src/ggml.c" -o /tmp/ggml.o
    gcc -O2 -I/tmp/ggml-inc -I"${LOCAL_REPO}/ggml/include" -I"${LOCAL_REPO}/ggml/src" \
        -c "${LOCAL_REPO}/ggml/src/ggml-quants.c" -o /tmp/ggml-quants.o
    g++ -O2 -std=c++17 -ffunction-sections -Wl,--gc-sections -Wall -Wextra -Wno-unused-function \
        -I/tmp/ggml-inc -I"${LOCAL_REPO}/ggml/include" -I"${LOCAL_REPO}/ggml/src" \
        "${LOCAL_REPO}/ggml/src/ggml-agx/tests/test_agx_repack.cpp" \
        "${LOCAL_REPO}/ggml/src/ggml-agx/agx-repack.cpp" \
        "${LOCAL_REPO}/ggml/src/ggml-threading.cpp" \
        /tmp/ggml.o /tmp/ggml-quants.o -lm -lpthread -o /tmp/test_agx_repack
    /tmp/test_agx_repack
}

step_glsl() {
    echo "[glsl] Validating all .comp shaders on ${REMOTE_HOST} with glslangValidator --spirv-val..."
    ssh_m4 "su - astaff -s /run/current-system/sw/bin/bash -c 'nix-shell /home/astaff/llama/shell.nix --run \"python3 ${REMOTE_REPO}/ggml/src/ggml-agx/tests/glsl_check.py\"'"
}

step_sync() {
    echo "[sync] Syncing ${LOCAL_REPO} -> ${REMOTE_HOST}:${REMOTE_REPO}..."
    ssh_m4 "mkdir -p ${REMOTE_REPO} && chown astaff:users ${REMOTE_REPO}"
    rsync -az --delete \
        --exclude='/.git' \
        --exclude='/build*' \
        --exclude='ggml/src/ggml-agx/formal/states' \
        -e "ssh -F ${SSH_CONF}" \
        "${LOCAL_REPO}/" "${REMOTE_HOST}:${REMOTE_REPO}/"
    ssh_m4 "chown -R astaff:users ${REMOTE_REPO}"
}

step_build() {
    local targets="${1:-test-backend-ops llama-cli llama-bench}"
    echo "[build] Building ggml-agx on ${REMOTE_HOST} (targets: ${targets})..."
    ssh_m4 "su - astaff -s /run/current-system/sw/bin/bash -c 'nix-shell /home/astaff/llama/shell.nix --run \"
        cmake -S ${REMOTE_REPO} -B ${REMOTE_BUILD} -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
            -DGGML_AGX=ON -DGGML_VULKAN=OFF -DGGML_CCACHE=OFF -DGGML_BLAS=OFF \
            -DGGML_NATIVE=OFF -DGGML_CPU_KLEIDIAI=OFF \
            -DGGML_CPU_ARM_ARCH=armv8.6-a+fp16+dotprod+i8mm+nosve+nosme \
            -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF && \
        cmake --build ${REMOTE_BUILD} --target ${targets} -j 8
    \"'"
}

step_roofline() {
    echo "[roofline] Building & running test_agx_roofline on ${REMOTE_HOST} via guarded-run.py..."
    ssh_m4 "su - astaff -s /run/current-system/sw/bin/bash -c 'nix-shell /home/astaff/llama/shell.nix --run \"
        g++ -O2 -std=c++17 -Wall -Wextra \
            -I${REMOTE_REPO}/ggml/include -I${REMOTE_REPO}/ggml/src \
            -I${REMOTE_REPO}/ggml/src/ggml-agx -I${REMOTE_REPO}/ggml/src/ggml-agx/kernels \
            ${REMOTE_REPO}/ggml/src/ggml-agx/tests/test_agx_roofline.cpp \
            ${REMOTE_BUILD}/bin/libggml-agx.so ${REMOTE_BUILD}/bin/libggml-base.so \
            -Wl,-rpath,${REMOTE_BUILD}/bin -ldl -lpthread \
            -o ${REMOTE_BUILD}/bin/test_agx_roofline
    \"'"
    run_guarded "roofline" "${REMOTE_BUILD}/bin/test_agx_roofline"
}

step_backend_ops() {
    echo "[backend-ops] Running test-backend-ops -b AGX0 via guarded-run.py..."
    run_guarded "backend-ops" "${REMOTE_BUILD}/bin/test-backend-ops -b AGX0"
}

step_bench() {
    echo "[bench] Running Inkling 2 Mini llama-bench on AGX0 via guarded-run.py..."
    run_guarded "bench-inkling2" \
        "${REMOTE_BUILD}/bin/llama-bench -m ${INKLING_MODEL} -dev AGX0 -ngl 99 -fa 1 -p 512 -n 128 -r 3"
}

case "${1:-all}" in
    tlc)         step_tlc ;;
    numerics)    step_local_numerics ;;
    repack)      step_local_repack ;;
    sync)        step_sync ;;
    glsl)        step_glsl ;;
    build)       shift; step_build "$*" ;;
    roofline)    step_roofline ;;
    backend-ops) step_backend_ops ;;
    bench)       step_bench ;;
    all)
        step_local_numerics
        step_local_repack
        step_tlc
        step_sync
        step_glsl
        step_build "test-backend-ops llama-cli llama-bench"
        step_roofline
        step_backend_ops
        ;;
    *)
        echo "Usage: $0 {tlc|numerics|repack|sync|glsl|build [targets]|roofline|backend-ops|bench|all}"
        exit 2
        ;;
esac
