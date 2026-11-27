#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_AGX_NAME        "AGX"
#define GGML_AGX_MAX_DEVICES 4

// Backend API (bare-metal Apple Silicon AGX3 via Linux drm_asahi)
GGML_BACKEND_API ggml_backend_t             ggml_backend_agx_init(size_t dev_num);
GGML_BACKEND_API bool                       ggml_backend_is_agx(ggml_backend_t backend);

GGML_BACKEND_API int                        ggml_backend_agx_get_device_count(void);
GGML_BACKEND_API void                       ggml_backend_agx_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void                       ggml_backend_agx_get_device_memory(int device, size_t * free, size_t * total);

// Default SLC-aware unified memory buffer type (AGX0)
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_agx_buffer_type(size_t dev_num);
// Repacked 128B cache-line-aligned super-tile weight buffer type (AGX0_Repack)
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_agx_repack_buffer_type(size_t dev_num);
// Pinned host-visible coherent buffer type
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_agx_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t         ggml_backend_agx_reg(void);

#ifdef __cplusplus
}
#endif
