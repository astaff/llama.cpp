#!/usr/bin/env bash
set -euo pipefail
REPO="/home/astaff/llama/llama-ggml-agx"
BUILD="${REPO}/build"
OBJS=("${BUILD}"/ggml/src/ggml-agx/CMakeFiles/ggml-agx.dir/*.o)

for t in test_agx_roofline test_agx_shaderdb_all test_agx_gemm_roofline; do
    g++ -O2 -std=c++17 -Wall -Wextra \
        -I"${REPO}/ggml/include" -I"${REPO}/ggml/src" \
        -I"${REPO}/ggml/src/ggml-agx" -I"${REPO}/ggml/src/ggml-agx/kernels" \
        "${REPO}/ggml/src/ggml-agx/tests/${t}.cpp" \
        "${OBJS[@]}" "${BUILD}/bin/libggml-base.so" \
        -Wl,-rpath,"${BUILD}/bin" -ldl -lpthread \
        -o "${BUILD}/bin/${t}"
    echo "BUILT ${BUILD}/bin/${t}"
done

for t in test_agx_diff_logits test_agx_bisect test_agx_profile; do
    g++ -O2 -std=c++17 -Wall -Wextra -Wno-deprecated-declarations \
        -I"${REPO}/include" -I"${REPO}/ggml/include" \
        "${REPO}/ggml/src/ggml-agx/tests/${t}.cpp" \
        "${BUILD}/bin/libllama.so" "${BUILD}/bin/libggml.so" "${BUILD}/bin/libggml-base.so" \
        -Wl,-rpath,"${BUILD}/bin" -lpthread \
        -o "${BUILD}/bin/${t}"
    echo "BUILT ${BUILD}/bin/${t}"
done
