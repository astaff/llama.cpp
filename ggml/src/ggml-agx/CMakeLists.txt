find_package(Threads REQUIRED)

set(GGML_AGX_SOURCES
    ggml-agx.cpp
    agx-common.h
    agx-drm.h
    agx-drm.cpp
    agx-cdm.h
    agx-cdm.cpp
    agx-compiler.h
    agx-compiler.cpp
)

# Modular Pillars 2, 3, 4/5: included automatically when present on disk so
# Pillar 1/4/6 builds cleanly standalone and seamlessly links all pillars once landed.
set(GGML_AGX_OPTIONAL_SOURCES
    agx-repack.cpp
    agx-gemm.cpp
    agx-inkling2.cpp
    agx-graph-exec.cpp
    agx-kv-hybrid.cpp
    agx-moe.cpp
)

foreach(_agx_src ${GGML_AGX_OPTIONAL_SOURCES})
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_agx_src}")
        list(APPEND GGML_AGX_SOURCES "${_agx_src}")
        message(STATUS "ggml-agx: including ${_agx_src}")
    endif()
endforeach()

ggml_add_backend_library(ggml-agx
    ../../include/ggml-agx.h
    ${GGML_AGX_SOURCES}
)

target_include_directories(ggml-agx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_compile_definitions(ggml-agx PRIVATE
    GGML_AGX_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)

if (GGML_AGX_DEBUG)
    target_compile_definitions(ggml-agx PRIVATE GGML_AGX_DEBUG=1)
endif()

target_link_libraries(ggml-agx PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
