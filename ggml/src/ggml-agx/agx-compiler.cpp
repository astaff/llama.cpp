#include "agx-compiler.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ============================================================================
// Minimal Self-Contained Vulkan 1.3 + EXT_shader_object + KHR_pipeline_exec ABI
// ============================================================================
// Loaded lazily via dlopen("libvulkan.so.1") ONLY when compiling a SPIR-V shader
// that misses the on-disk `.agxbin` cache. Zero compile-time Vulkan header dependency
// and zero Vulkan runtime involvement during CDM compute submission!

using AgxVkBool32 = uint32_t;
using AgxVkFlags  = uint32_t;
using AgxVkResult = int32_t;

struct AgxVkInstance_T;
struct AgxVkPhysicalDevice_T;
struct AgxVkDevice_T;
struct AgxVkShaderModule_T;
struct AgxVkPipelineLayout_T;
struct AgxVkPipelineCache_T;
struct AgxVkPipeline_T;
struct AgxVkShaderEXT_T;

using AgxVkInstance       = AgxVkInstance_T *;
using AgxVkPhysicalDevice = AgxVkPhysicalDevice_T *;
using AgxVkDevice         = AgxVkDevice_T *;
using AgxVkShaderModule   = AgxVkShaderModule_T *;
using AgxVkPipelineLayout = AgxVkPipelineLayout_T *;
using AgxVkPipelineCache  = AgxVkPipelineCache_T *;
using AgxVkPipeline       = AgxVkPipeline_T *;
using AgxVkShaderEXT      = AgxVkShaderEXT_T *;

constexpr AgxVkResult AGX_VK_SUCCESS = 0;
constexpr AgxVkBool32 AGX_VK_TRUE    = 1;
constexpr AgxVkBool32 AGX_VK_FALSE   = 0;

constexpr uint32_t AGX_VK_API_VERSION_1_3 = (1u << 22) | (3u << 12);

constexpr uint32_t AGX_VK_STYPE_APPLICATION_INFO                                       = 0;
constexpr uint32_t AGX_VK_STYPE_INSTANCE_CREATE_INFO                                   = 1;
constexpr uint32_t AGX_VK_STYPE_DEVICE_QUEUE_CREATE_INFO                               = 2;
constexpr uint32_t AGX_VK_STYPE_DEVICE_CREATE_INFO                                     = 3;
constexpr uint32_t AGX_VK_STYPE_SHADER_MODULE_CREATE_INFO                              = 16;
constexpr uint32_t AGX_VK_STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO                      = 18;
constexpr uint32_t AGX_VK_STYPE_COMPUTE_PIPELINE_CREATE_INFO                           = 29;
constexpr uint32_t AGX_VK_STYPE_PIPELINE_LAYOUT_CREATE_INFO                            = 30;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_FEATURES_2                             = 1000059000;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES                    = 49;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES                    = 51;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES                    = 53;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEAT    = 1000269000;
constexpr uint32_t AGX_VK_STYPE_PIPELINE_INFO_KHR                                      = 1000269001;
constexpr uint32_t AGX_VK_STYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR                     = 1000269002;
constexpr uint32_t AGX_VK_STYPE_PIPELINE_EXECUTABLE_INFO_KHR                           = 1000269003;
constexpr uint32_t AGX_VK_STYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR                      = 1000269004;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT             = 1000482000;
constexpr uint32_t AGX_VK_STYPE_SHADER_CREATE_INFO_EXT                                 = 1000482002;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR        = 1000506000;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR    = 1000416000;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR   = 1000528000;
constexpr uint32_t AGX_VK_STYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR           = 1000141000;

constexpr AgxVkFlags AGX_VK_SHADER_STAGE_COMPUTE_BIT                      = 0x00000020u;
constexpr AgxVkFlags AGX_VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR    = 0x00000040u;
constexpr uint32_t   AGX_VK_SHADER_CODE_TYPE_SPIRV_EXT                    = 1u;

constexpr uint32_t AGX_VK_STAT_FMT_BOOL32  = 0;
constexpr uint32_t AGX_VK_STAT_FMT_INT64   = 1;
constexpr uint32_t AGX_VK_STAT_FMT_UINT64  = 2;
constexpr uint32_t AGX_VK_STAT_FMT_FLOAT64 = 3;

struct AgxVkApplicationInfo {
    uint32_t     sType;
    const void * pNext;
    const char * pApplicationName;
    uint32_t     applicationVersion;
    const char * pEngineName;
    uint32_t     engineVersion;
    uint32_t     apiVersion;
};

struct AgxVkInstanceCreateInfo {
    uint32_t                     sType;
    const void *                 pNext;
    AgxVkFlags                   flags;
    const AgxVkApplicationInfo * pApplicationInfo;
    uint32_t                     enabledLayerCount;
    const char * const *         ppEnabledLayerNames;
    uint32_t                     enabledExtensionCount;
    const char * const *         ppEnabledExtensionNames;
};

struct AgxVkExtensionProperties {
    char     extensionName[256];
    uint32_t specVersion;
};

struct AgxVkPhysicalDeviceFeatures {
    AgxVkBool32 raw[55]; // exact 55 VkBool32 fields in Vulkan 1.0 VkPhysicalDeviceFeatures
};
// Field indices in VkPhysicalDeviceFeatures:
// 52 = shaderInt64, 53 = shaderInt16, 51 = shaderFloat64
constexpr size_t AGX_VK_FEAT_SHADER_INT64 = 52;
constexpr size_t AGX_VK_FEAT_SHADER_INT16 = 53;

struct AgxVkPhysicalDeviceFeatures2 {
    uint32_t                    sType;
    void *                      pNext;
    AgxVkPhysicalDeviceFeatures features;
};

struct AgxVkPhysicalDeviceVulkan11Features {
    uint32_t    sType;
    void *      pNext;
    AgxVkBool32 storageBuffer16BitAccess;
    AgxVkBool32 uniformAndStorageBuffer16BitAccess;
    AgxVkBool32 storagePushConstant16;
    AgxVkBool32 storageInputOutput16;
    AgxVkBool32 multiview;
    AgxVkBool32 multiviewGeometryShader;
    AgxVkBool32 multiviewTessellationShader;
    AgxVkBool32 variablePointersStorageBuffer;
    AgxVkBool32 variablePointers;
    AgxVkBool32 protectedMemory;
    AgxVkBool32 samplerYcbcrConversion;
    AgxVkBool32 shaderDrawParameters;
};

struct AgxVkPhysicalDeviceVulkan12Features {
    uint32_t    sType;
    void *      pNext;
    AgxVkBool32 samplerMirrorClampToEdge;
    AgxVkBool32 drawIndirectCount;
    AgxVkBool32 storageBuffer8BitAccess;
    AgxVkBool32 uniformAndStorageBuffer8BitAccess;
    AgxVkBool32 storagePushConstant8;
    AgxVkBool32 shaderBufferInt64Atomics;
    AgxVkBool32 shaderSharedInt64Atomics;
    AgxVkBool32 shaderFloat16;
    AgxVkBool32 shaderInt8;
    AgxVkBool32 descriptorIndexing;
    AgxVkBool32 shaderInputAttachmentArrayDynamicIndexing;
    AgxVkBool32 shaderUniformTexelBufferArrayDynamicIndexing;
    AgxVkBool32 shaderStorageTexelBufferArrayDynamicIndexing;
    AgxVkBool32 shaderUniformBufferArrayNonUniformIndexing;
    AgxVkBool32 shaderSampledImageArrayNonUniformIndexing;
    AgxVkBool32 shaderStorageBufferArrayNonUniformIndexing;
    AgxVkBool32 shaderStorageImageArrayNonUniformIndexing;
    AgxVkBool32 shaderInputAttachmentArrayNonUniformIndexing;
    AgxVkBool32 shaderUniformTexelBufferArrayNonUniformIndexing;
    AgxVkBool32 shaderStorageTexelBufferArrayNonUniformIndexing;
    AgxVkBool32 descriptorBindingUniformBufferUpdateAfterBind;
    AgxVkBool32 descriptorBindingSampledImageUpdateAfterBind;
    AgxVkBool32 descriptorBindingStorageImageUpdateAfterBind;
    AgxVkBool32 descriptorBindingStorageBufferUpdateAfterBind;
    AgxVkBool32 descriptorBindingUniformTexelBufferUpdateAfterBind;
    AgxVkBool32 descriptorBindingStorageTexelBufferUpdateAfterBind;
    AgxVkBool32 descriptorBindingUpdateUnusedWhilePending;
    AgxVkBool32 descriptorBindingPartiallyBound;
    AgxVkBool32 descriptorBindingVariableDescriptorCount;
    AgxVkBool32 runtimeDescriptorArray;
    AgxVkBool32 samplerFilterMinmax;
    AgxVkBool32 scalarBlockLayout;
    AgxVkBool32 imagelessFramebuffer;
    AgxVkBool32 uniformBufferStandardLayout;
    AgxVkBool32 shaderSubgroupExtendedTypes;
    AgxVkBool32 separateDepthStencilLayouts;
    AgxVkBool32 hostQueryReset;
    AgxVkBool32 timelineSemaphore;
    AgxVkBool32 bufferDeviceAddress;
    AgxVkBool32 bufferDeviceAddressCaptureReplay;
    AgxVkBool32 bufferDeviceAddressMultiDevice;
    AgxVkBool32 vulkanMemoryModel;
    AgxVkBool32 vulkanMemoryModelDeviceScope;
    AgxVkBool32 vulkanMemoryModelAvailabilityVisibilityChains;
    AgxVkBool32 shaderOutputViewportIndex;
    AgxVkBool32 shaderOutputLayer;
    AgxVkBool32 subgroupBroadcastDynamicId;
};

struct AgxVkPhysicalDeviceVulkan13Features {
    uint32_t    sType;
    void *      pNext;
    AgxVkBool32 robustImageAccess;
    AgxVkBool32 inlineUniformBlock;
    AgxVkBool32 descriptorBindingInlineUniformBlockUpdateAfterBind;
    AgxVkBool32 pipelineCreationCacheControl;
    AgxVkBool32 privateData;
    AgxVkBool32 shaderDemoteToHelperInvocation;
    AgxVkBool32 shaderTerminateInvocation;
    AgxVkBool32 subgroupSizeControl;
    AgxVkBool32 computeFullSubgroups;
    AgxVkBool32 synchronization2;
    AgxVkBool32 textureCompressionASTC_HDR;
    AgxVkBool32 shaderZeroInitializeWorkgroupMemory;
    AgxVkBool32 dynamicRendering;
    AgxVkBool32 shaderIntegerDotProduct;
    AgxVkBool32 maintenance4;
};

struct AgxVkPhysicalDeviceShaderObjectFeaturesEXT {
    uint32_t    sType;
    void *      pNext;
    AgxVkBool32 shaderObject;
};

struct AgxVkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR {
    uint32_t    sType;
    void *      pNext;
    AgxVkBool32 pipelineExecutableInfo;
};

struct AgxVkPhysicalDeviceCooperativeMatrixFeaturesKHR {
    uint32_t    sType;
    void *      pNext;
    AgxVkBool32 cooperativeMatrix;
    AgxVkBool32 cooperativeMatrixRobustBufferAccess;
};

struct AgxVkPhysicalDeviceShaderBfloat16FeaturesKHR {
    uint32_t    sType;
    void *      pNext;
    AgxVkBool32 shaderBFloat16Type;
    AgxVkBool32 shaderBFloat16DotProduct;
    AgxVkBool32 shaderBFloat16CooperativeMatrix;
};

struct AgxVkDeviceQueueCreateInfo {
    uint32_t      sType;
    const void *  pNext;
    AgxVkFlags    flags;
    uint32_t      queueFamilyIndex;
    uint32_t      queueCount;
    const float * pQueuePriorities;
};

struct AgxVkDeviceCreateInfo {
    uint32_t                           sType;
    const void *                       pNext;
    AgxVkFlags                         flags;
    uint32_t                           queueCreateInfoCount;
    const AgxVkDeviceQueueCreateInfo * pQueueCreateInfos;
    uint32_t                           enabledLayerCount;
    const char * const *               ppEnabledLayerNames;
    uint32_t                           enabledExtensionCount;
    const char * const *               ppEnabledExtensionNames;
    const void *                       pEnabledFeatures;
};

struct AgxVkPushConstantRange {
    AgxVkFlags stageFlags;
    uint32_t   offset;
    uint32_t   size;
};

struct AgxVkShaderCreateInfoEXT {
    uint32_t                       sType;
    const void *                   pNext;
    AgxVkFlags                     flags;
    uint32_t                       stage;
    uint32_t                       nextStage;
    uint32_t                       codeType;
    size_t                         codeSize;
    const void *                   pCode;
    const char *                   pName;
    uint32_t                       setLayoutCount;
    const void *                   pSetLayouts;
    uint32_t                       pushConstantRangeCount;
    const AgxVkPushConstantRange * pPushConstantRanges;
    const void *                   pSpecializationInfo;
};

struct AgxVkShaderModuleCreateInfo {
    uint32_t         sType;
    const void *     pNext;
    AgxVkFlags       flags;
    size_t           codeSize;
    const uint32_t * pCode;
};

struct AgxVkPipelineLayoutCreateInfo {
    uint32_t                       sType;
    const void *                   pNext;
    AgxVkFlags                     flags;
    uint32_t                       setLayoutCount;
    const void *                   pSetLayouts;
    uint32_t                       pushConstantRangeCount;
    const AgxVkPushConstantRange * pPushConstantRanges;
};

struct AgxVkPipelineShaderStageCreateInfo {
    uint32_t            sType;
    const void *        pNext;
    AgxVkFlags          flags;
    uint32_t            stage;
    AgxVkShaderModule   module;
    const char *        pName;
    const void *        pSpecializationInfo;
};

struct AgxVkComputePipelineCreateInfo {
    uint32_t                           sType;
    const void *                       pNext;
    AgxVkFlags                         flags;
    AgxVkPipelineShaderStageCreateInfo stage;
    AgxVkPipelineLayout                layout;
    AgxVkPipeline                      basePipelineHandle;
    int32_t                            basePipelineIndex;
};

struct AgxVkPipelineInfoKHR {
    uint32_t      sType;
    const void *  pNext;
    AgxVkPipeline pipeline;
};

struct AgxVkPipelineExecutablePropertiesKHR {
    uint32_t   sType;
    void *     pNext;
    AgxVkFlags stages;
    char       name[256];
    char       description[256];
    uint32_t   subgroupSize;
};

struct AgxVkPipelineExecutableInfoKHR {
    uint32_t      sType;
    const void *  pNext;
    AgxVkPipeline pipeline;
    uint32_t      executableIndex;
};

union AgxVkPipelineExecutableStatisticValueKHR {
    AgxVkBool32 b32;
    int64_t     i64;
    uint64_t    u64;
    double      f64;
};

struct AgxVkPipelineExecutableStatisticKHR {
    uint32_t                                 sType;
    void *                                   pNext;
    char                                     name[256];
    char                                     description[256];
    uint32_t                                 format;
    AgxVkPipelineExecutableStatisticValueKHR value;
};

using PFN_agx_vkGetInstanceProcAddr = void * (*)(AgxVkInstance, const char *);
using PFN_agx_vkGetDeviceProcAddr   = void * (*)(AgxVkDevice, const char *);
using PFN_agx_vkCreateInstance      = AgxVkResult (*)(const AgxVkInstanceCreateInfo *, const void *, AgxVkInstance *);
using PFN_agx_vkDestroyInstance     = void (*)(AgxVkInstance, const void *);
using PFN_agx_vkEnumeratePhysicalDevices = AgxVkResult (*)(AgxVkInstance, uint32_t *, AgxVkPhysicalDevice *);
using PFN_agx_vkEnumerateDeviceExtensionProperties = AgxVkResult (*)(AgxVkPhysicalDevice, const char *, uint32_t *, AgxVkExtensionProperties *);
using PFN_agx_vkGetPhysicalDeviceFeatures2 = void (*)(AgxVkPhysicalDevice, AgxVkPhysicalDeviceFeatures2 *);
using PFN_agx_vkCreateDevice        = AgxVkResult (*)(AgxVkPhysicalDevice, const AgxVkDeviceCreateInfo *, const void *, AgxVkDevice *);
using PFN_agx_vkDestroyDevice       = void (*)(AgxVkDevice, const void *);
using PFN_agx_vkCreateShadersEXT    = AgxVkResult (*)(AgxVkDevice, uint32_t, const AgxVkShaderCreateInfoEXT *, const void *, AgxVkShaderEXT *);
using PFN_agx_vkGetShaderBinaryDataEXT = AgxVkResult (*)(AgxVkDevice, AgxVkShaderEXT, size_t *, void *);
using PFN_agx_vkDestroyShaderEXT    = void (*)(AgxVkDevice, AgxVkShaderEXT, const void *);
using PFN_agx_vkCreateShaderModule  = AgxVkResult (*)(AgxVkDevice, const AgxVkShaderModuleCreateInfo *, const void *, AgxVkShaderModule *);
using PFN_agx_vkDestroyShaderModule = void (*)(AgxVkDevice, AgxVkShaderModule, const void *);
using PFN_agx_vkCreatePipelineLayout = AgxVkResult (*)(AgxVkDevice, const AgxVkPipelineLayoutCreateInfo *, const void *, AgxVkPipelineLayout *);
using PFN_agx_vkDestroyPipelineLayout = void (*)(AgxVkDevice, AgxVkPipelineLayout, const void *);
using PFN_agx_vkCreateComputePipelines = AgxVkResult (*)(AgxVkDevice, AgxVkPipelineCache, uint32_t, const AgxVkComputePipelineCreateInfo *, const void *, AgxVkPipeline *);
using PFN_agx_vkDestroyPipeline     = void (*)(AgxVkDevice, AgxVkPipeline, const void *);
using PFN_agx_vkGetPipelineExecutablePropertiesKHR = AgxVkResult (*)(AgxVkDevice, const AgxVkPipelineInfoKHR *, uint32_t *, AgxVkPipelineExecutablePropertiesKHR *);
using PFN_agx_vkGetPipelineExecutableStatisticsKHR = AgxVkResult (*)(AgxVkDevice, const AgxVkPipelineExecutableInfoKHR *, uint32_t *, AgxVkPipelineExecutableStatisticKHR *);

// ============================================================================
// Binary & SPIR-V Helpers
// ============================================================================

static inline uint16_t rd_u16(const uint8_t * p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint32_t rd_u32(const uint8_t * p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static uint64_t fnv1a_64(const void * data, size_t n, uint64_t seed = 0xcbf29ce484222325ULL) {
    const auto * b = static_cast<const uint8_t *>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static std::string compute_cache_key(const uint32_t * spv, size_t word_count,
                                     const agx_compile_opts & opts,
                                     const agx_drm_device_info & info) {
    uint64_t h1 = fnv1a_64(spv, word_count * sizeof(uint32_t), 0xcbf29ce484222325ULL);
    uint64_t h2 = fnv1a_64(spv, word_count * sizeof(uint32_t), 0x84222325cbf29ce4ULL);
    const char * ep = opts.entry_point ? opts.entry_point : "main";
    h1 = fnv1a_64(ep, std::strlen(ep), h1);
    h2 = fnv1a_64(&opts.push_constant_bytes, sizeof(opts.push_constant_bytes), h2);
    uint32_t meta[4] = {
        info.chip_id,
        info.gpu_hal_generation,
        opts.allow_preamble ? 1u : 0u,
        0x41475831u, // cache schema v1
    };
    h1 = fnv1a_64(meta, sizeof(meta), h1);
    h2 = fnv1a_64(meta, sizeof(meta), h2);

    char buf[48];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64 "%016" PRIx64, h1, h2);
    return std::string(buf);
}

static void parse_spirv_local_size(const uint32_t * words, size_t count, uint32_t out_local[3]) {
    out_local[0] = 32;
    out_local[1] = 1;
    out_local[2] = 1;
    if (!words || count < 5 || words[0] != 0x07230203u) {
        return;
    }
    size_t i = 5;
    while (i < count) {
        const uint32_t hdr = words[i];
        const uint32_t wc  = hdr >> 16;
        const uint32_t op  = hdr & 0xffffu;
        if (wc == 0 || i + wc > count) {
            break;
        }
        // OpExecutionMode = 16, ExecutionModeLocalSize = 17
        if (op == 16u && wc >= 6u && words[i + 2] == 17u) {
            out_local[0] = std::max(1u, words[i + 3]);
            out_local[1] = std::max(1u, words[i + 4]);
            out_local[2] = std::max(1u, words[i + 5]);
            return;
        }
        i += wc;
    }
}

static std::string find_glsl_compiler(bool * out_is_glslc) {
    if (const char * env = std::getenv("GGML_AGX_GLSLANG"); env && env[0]) {
        if (out_is_glslc) *out_is_glslc = (std::strstr(env, "glslc") != nullptr);
        return env;
    }
    // Check PATH for glslangValidator first, then glslc
    const char * path_env = std::getenv("PATH");
    if (path_env) {
        std::stringstream ss(path_env);
        std::string dir;
        while (std::getline(ss, dir, ':')) {
            if (dir.empty()) continue;
            fs::path p1 = fs::path(dir) / "glslangValidator";
            if (::access(p1.c_str(), X_OK) == 0) {
                if (out_is_glslc) *out_is_glslc = false;
                return p1.string();
            }
        }
        ss.clear();
        ss.str(path_env);
        while (std::getline(ss, dir, ':')) {
            if (dir.empty()) continue;
            fs::path p2 = fs::path(dir) / "glslc";
            if (::access(p2.c_str(), X_OK) == 0) {
                if (out_is_glslc) *out_is_glslc = true;
                return p2.string();
            }
        }
    }
    if (out_is_glslc) *out_is_glslc = false;
    return "glslangValidator";
}

static bool run_glsl_to_spirv(const std::string & glsl_source,
                              const std::string & default_kernel_dir,
                              const agx_compile_opts & opts,
                              std::vector<uint32_t> & out_spv,
                              std::string & out_err) {
    static std::atomic<uint64_t> seq{0};
    const uint64_t id = seq.fetch_add(1, std::memory_order_relaxed);
    const std::string base = "/tmp/ggml_agx_" + std::to_string(::getpid()) + "_" + std::to_string(id);
    const std::string src_path = base + ".comp";
    const std::string spv_path = base + ".spv";
    const std::string log_path = base + ".log";

    {
        std::ofstream ofs(src_path, std::ios::binary);
        if (!ofs) {
            out_err = "Failed to write temp GLSL file: " + src_path;
            return false;
        }
        ofs.write(glsl_source.data(), static_cast<std::streamsize>(glsl_source.size()));
    }

    bool is_glslc = false;
    const std::string tool = find_glsl_compiler(&is_glslc);

    std::vector<std::string> argv_str;
    argv_str.push_back(tool);
    if (is_glslc) {
        argv_str.push_back("--target-env=vulkan1.3");
        argv_str.push_back("-O");
        argv_str.push_back("-fshader-stage=compute");
        if (opts.entry_point && opts.entry_point[0] && std::strcmp(opts.entry_point, "main") != 0) {
            argv_str.push_back(std::string("-fentry-point=") + opts.entry_point);
        }
    } else {
        argv_str.push_back("-V");
        argv_str.push_back("--target-env");
        argv_str.push_back("vulkan1.3");
        if (opts.entry_point && opts.entry_point[0] && std::strcmp(opts.entry_point, "main") != 0) {
            argv_str.push_back("-e");
            argv_str.push_back(opts.entry_point);
        }
    }

    if (!default_kernel_dir.empty()) {
        argv_str.push_back("-I" + default_kernel_dir);
    }
    for (const auto & inc : opts.include_dirs) {
        if (!inc.empty()) {
            argv_str.push_back("-I" + inc);
        }
    }
    for (const auto & def : opts.defines) {
        if (!def.empty()) {
            argv_str.push_back("-D" + def);
        }
    }
    argv_str.push_back(src_path);
    argv_str.push_back("-o");
    argv_str.push_back(spv_path);

    std::vector<char *> argv;
    argv.reserve(argv_str.size() + 1);
    for (auto & s : argv_str) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);

    pid_t pid = ::fork();
    if (pid < 0) {
        out_err = "fork() failed running GLSL compiler";
        std::remove(src_path.c_str());
        return false;
    }
    if (pid == 0) {
        FILE * f = std::fopen(log_path.c_str(), "w");
        if (f) {
            ::dup2(fileno(f), STDOUT_FILENO);
            ::dup2(fileno(f), STDERR_FILENO);
            std::fclose(f);
        }
        ::execvp(argv[0], argv.data());
        _exit(127);
    }

    int status = 0;
    ::waitpid(pid, &status, 0);

    std::string log_txt;
    {
        std::ifstream lif(log_path);
        if (lif) {
            std::stringstream ss;
            ss << lif.rdbuf();
            log_txt = ss.str();
        }
    }

    bool ok = false;
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        std::ifstream sif(spv_path, std::ios::binary | std::ios::ate);
        if (sif) {
            const std::streamsize sz = sif.tellg();
            if (sz > 0 && (sz % 4) == 0) {
                sif.seekg(0);
                out_spv.resize(static_cast<size_t>(sz) / 4u);
                sif.read(reinterpret_cast<char *>(out_spv.data()), sz);
                ok = true;
            }
        }
    }
    if (!ok) {
        out_err = "GLSL compilation failed (" + tool + "): " + log_txt;
    }

    std::remove(src_path.c_str());
    std::remove(spv_path.c_str());
    std::remove(log_path.c_str());
    return ok;
}

// ============================================================================
// Borealis Serialized Binary Layout (Self-Calibrated at Runtime!)
// ============================================================================

struct agx_borealis_abi {
    bool     calibrated         = false;
    uint32_t push_offset_bytes  = 0x338u; // empirically verified on ca90d8c; recalibrated live
    // Relative offsets inside `struct agx_shader_info`:
    size_t   dma_count_rel      = 0x459u;
    size_t   dma_array_rel      = 0x45cu;
    size_t   local_size_rel     = 0x4bcu; // uint16_t threadgroup SRAM bytes
    size_t   scratch_size_rel   = 0x4c0u; // uint32_t scratch_size
    size_t   spectpr_rel        = 0x4c8u; // uint32_t agx3_spectpr_scratch_size
    size_t   main_size_rel      = 0x4ccu; // uint32_t main_size
    size_t   preamble_size_rel  = 0x4d0u; // uint32_t preamble_size
    size_t   has_preamble_rel   = 0x4d4u; // uint8_t  has_preamble
    size_t   nr_gprs_rel        = 0x53eu; // uint16_t nr_gprs
    size_t   staging_halfs_rel  = 0x542u; // uint16_t agx3_staging_halfs
    size_t   staging_known_rel  = 0x544u; // uint8_t  agx3_staging_known
    size_t   wg_size_rel        = 0x558u; // uint16_t workgroup_size[3]
    size_t   rodata_rel         = 0x564u; // struct agx_rodata (offset, base_uniform, size_16)
    size_t   stats_rel          = 0x56cu; // struct agx2_stats
    // Offset of hk_shader_info::agx3_uses_root relative to info_off (negative, before root_descriptor_size)
    int64_t  uses_root_delta    = 0;
    int64_t  root_uniform_delta = 0;
    bool     has_hk_root_deltas = false;
};

struct agx_borealis_vk_ctx {
    void *               dl_handle = nullptr;
    AgxVkInstance        instance  = nullptr;
    AgxVkPhysicalDevice  phys_dev  = nullptr;
    AgxVkDevice          device    = nullptr;

    PFN_agx_vkDestroyInstance                   destroy_instance        = nullptr;
    PFN_agx_vkDestroyDevice                     destroy_device          = nullptr;
    PFN_agx_vkCreateShadersEXT                  create_shaders_ext      = nullptr;
    PFN_agx_vkGetShaderBinaryDataEXT            get_shader_bin_ext      = nullptr;
    PFN_agx_vkDestroyShaderEXT                  destroy_shader_ext      = nullptr;
    PFN_agx_vkCreateShaderModule                create_shader_module    = nullptr;
    PFN_agx_vkDestroyShaderModule               destroy_shader_module   = nullptr;
    PFN_agx_vkCreatePipelineLayout              create_pipeline_layout  = nullptr;
    PFN_agx_vkDestroyPipelineLayout             destroy_pipeline_layout = nullptr;
    PFN_agx_vkCreateComputePipelines            create_compute_pipes    = nullptr;
    PFN_agx_vkDestroyPipeline                   destroy_pipeline        = nullptr;
    PFN_agx_vkGetPipelineExecutablePropertiesKHR get_pipe_exec_props    = nullptr;
    PFN_agx_vkGetPipelineExecutableStatisticsKHR get_pipe_exec_stats    = nullptr;

    agx_borealis_abi abi;
    bool             has_coopmat = false;
    bool             initialized = false;
    bool             failed      = false;
};

static bool extract_pipe_stats(agx_borealis_vk_ctx & vk,
                               const uint32_t * spv,
                               size_t word_count,
                               const char * entry_point,
                               uint32_t push_bytes,
                               agx_shader_stats & out_stats) {
    if (!vk.create_shader_module || !vk.create_compute_pipes || !vk.get_pipe_exec_stats) {
        return false;
    }

    AgxVkShaderModuleCreateInfo smci{};
    smci.sType    = AGX_VK_STYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = word_count * sizeof(uint32_t);
    smci.pCode    = spv;
    AgxVkShaderModule sm = nullptr;
    if (vk.create_shader_module(vk.device, &smci, nullptr, &sm) != AGX_VK_SUCCESS || !sm) {
        return false;
    }

    AgxVkPushConstantRange pcr{ AGX_VK_SHADER_STAGE_COMPUTE_BIT, 0u, std::min<uint32_t>(push_bytes, AGX_MAX_PUSH_BYTES) };
    AgxVkPipelineLayoutCreateInfo plci{};
    plci.sType = AGX_VK_STYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (pcr.size > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
    }
    AgxVkPipelineLayout pl = nullptr;
    if (vk.create_pipeline_layout(vk.device, &plci, nullptr, &pl) != AGX_VK_SUCCESS || !pl) {
        vk.destroy_shader_module(vk.device, sm, nullptr);
        return false;
    }

    AgxVkComputePipelineCreateInfo cpci{};
    cpci.sType       = AGX_VK_STYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.flags       = AGX_VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    cpci.stage.sType = AGX_VK_STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = AGX_VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName  = entry_point ? entry_point : "main";
    cpci.layout       = pl;

    AgxVkPipeline pipe = nullptr;
    AgxVkResult r = vk.create_compute_pipes(vk.device, nullptr, 1, &cpci, nullptr, &pipe);
    bool got = false;
    if (r == AGX_VK_SUCCESS && pipe) {
        AgxVkPipelineInfoKHR pi{ AGX_VK_STYPE_PIPELINE_INFO_KHR, nullptr, pipe };
        uint32_t ecnt = 0;
        vk.get_pipe_exec_props(vk.device, &pi, &ecnt, nullptr);
        if (ecnt > 0) {
            AgxVkPipelineExecutableInfoKHR ei{ AGX_VK_STYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pipe, 0u };
            uint32_t scnt = 0;
            vk.get_pipe_exec_stats(vk.device, &ei, &scnt, nullptr);
            std::vector<AgxVkPipelineExecutableStatisticKHR> stats(scnt);
            for (auto & s : stats) {
                s.sType = AGX_VK_STYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            }
            if (scnt > 0 && vk.get_pipe_exec_stats(vk.device, &ei, &scnt, stats.data()) == AGX_VK_SUCCESS) {
                got = true;
                out_stats.stats_valid = true;
                for (const auto & s : stats) {
                    const uint64_t val = (s.format == AGX_VK_STAT_FMT_UINT64) ? s.value.u64 :
                                         (s.format == AGX_VK_STAT_FMT_INT64)  ? static_cast<uint64_t>(s.value.i64) : 0ULL;
                    if (std::strcmp(s.name, "GPRs") == 0) {
                        out_stats.nr_gprs_16 = static_cast<uint32_t>(val);
                        out_stats.nr_gprs_32 = (out_stats.nr_gprs_16 + 1u) / 2u;
                    } else if (std::strcmp(s.name, "Spills") == 0) {
                        out_stats.spills = static_cast<uint32_t>(val);
                    } else if (std::strcmp(s.name, "Fills") == 0) {
                        out_stats.fills = static_cast<uint32_t>(val);
                    } else if (std::strcmp(s.name, "Instructions") == 0) {
                        out_stats.instructions = static_cast<uint32_t>(val);
                    } else if (std::strcmp(s.name, "Code size") == 0) {
                        out_stats.code_bytes = static_cast<uint32_t>(val);
                    } else if (std::strcmp(s.name, "Scratch") == 0) {
                        out_stats.scratch_bytes = static_cast<uint32_t>(val);
                    } else if (std::strcmp(s.name, "Threads") == 0) {
                        out_stats.max_simdgroups_per_core = static_cast<uint32_t>(val / AGX_SIMD_WIDTH);
                    }
                }
            }
        }
        vk.destroy_pipeline(vk.device, pipe, nullptr);
    }

    vk.destroy_pipeline_layout(vk.device, pl, nullptr);
    vk.destroy_shader_module(vk.device, sm, nullptr);
    return got;
}

static bool extract_shader_ext_blob(agx_borealis_vk_ctx & vk,
                                    const uint32_t * spv,
                                    size_t word_count,
                                    const char * entry_point,
                                    uint32_t push_bytes,
                                    std::vector<uint8_t> & out_blob) {
    if (!vk.create_shaders_ext || !vk.get_shader_bin_ext) {
        return false;
    }
    AgxVkPushConstantRange pcr{ AGX_VK_SHADER_STAGE_COMPUTE_BIT, 0u, std::min<uint32_t>(push_bytes, AGX_MAX_PUSH_BYTES) };
    AgxVkShaderCreateInfoEXT sci{};
    sci.sType    = AGX_VK_STYPE_SHADER_CREATE_INFO_EXT;
    sci.stage    = AGX_VK_SHADER_STAGE_COMPUTE_BIT;
    sci.codeType = AGX_VK_SHADER_CODE_TYPE_SPIRV_EXT;
    sci.codeSize = word_count * sizeof(uint32_t);
    sci.pCode    = spv;
    sci.pName    = entry_point ? entry_point : "main";
    if (pcr.size > 0) {
        sci.pushConstantRangeCount = 1;
        sci.pPushConstantRanges    = &pcr;
    }

    AgxVkShaderEXT shader = nullptr;
    if (vk.create_shaders_ext(vk.device, 1, &sci, nullptr, &shader) != AGX_VK_SUCCESS || !shader) {
        return false;
    }

    size_t sz = 0;
    AgxVkResult r = vk.get_shader_bin_ext(vk.device, shader, &sz, nullptr);
    bool ok = false;
    if (r == AGX_VK_SUCCESS && sz > 64) {
        out_blob.resize(sz);
        if (vk.get_shader_bin_ext(vk.device, shader, &sz, out_blob.data()) == AGX_VK_SUCCESS) {
            ok = true;
        }
    }
    vk.destroy_shader_ext(vk.device, shader, nullptr);
    return ok;
}

// Locate the tail `[uint32 code_size][uint32 data_size][code...][data...]` of
// `vkGetShaderBinaryDataEXT` (produced by `hk_shader_serialize` in Borealis).
static bool locate_code_tail(const std::vector<uint8_t> & blob,
                             uint32_t expected_code_size,
                             size_t & out_hdr_off,
                             uint32_t & out_code_size,
                             uint32_t & out_data_size) {
    const size_t n = blob.size();
    if (n < 64) {
        return false;
    }
    // Scan backward from n - 8 for `[code_size, data_size]` where
    // `off + 8 + code_size + data_size == n`.
    // Prefer an exact match with `expected_code_size` (from VK_KHR_pipeline_executable_properties)
    // or a plausible code_size (>= 4, multiple of 2, ending with AGX3 STOP `0e 00 00 00` or near end).
    size_t fallback_off = 0;
    uint32_t fallback_cs = 0, fallback_ds = 0;
    for (size_t off = n - 8; off >= 64; --off) {
        const uint32_t cs = rd_u32(blob.data() + off);
        const uint32_t ds = rd_u32(blob.data() + off + 4);
        if (cs >= 4u && cs <= n && ds <= n && (off + 8u + cs + ds == n)) {
            if (expected_code_size != 0 && cs == expected_code_size) {
                out_hdr_off   = off;
                out_code_size = cs;
                out_data_size = ds;
                return true;
            }
            if (fallback_off == 0 && (cs % 2u == 0u)) {
                fallback_off = off;
                fallback_cs  = cs;
                fallback_ds  = ds;
            }
        }
        if (off == 64) break;
    }
    if (fallback_off != 0) {
        out_hdr_off   = fallback_off;
        out_code_size = fallback_cs;
        out_data_size = fallback_ds;
        return true;
    }
    return false;
}

// Locate `struct agx_shader_info` start inside `blob` before `code_hdr_off`.
// `agx_shader_info` starts with `{ uint32_t stage == 5 (MESA_SHADER_COMPUTE), uint32_t binary_size == code_size }`.
static bool locate_agx_shader_info(const std::vector<uint8_t> & blob,
                                   size_t code_hdr_off,
                                   uint32_t code_size,
                                   size_t & out_info_off) {
    // Search backward from code_hdr_off (agx_shader_info is ~1500..2200 bytes long).
    const size_t min_off = (code_hdr_off > 4096u) ? (code_hdr_off - 4096u) : 64u;
    for (size_t off = code_hdr_off - 8u; off >= min_off; off -= 4u) {
        const uint32_t stg = rd_u32(blob.data() + off);
        const uint32_t bsz = rd_u32(blob.data() + off + 4u);
        if (stg == 5u && bsz == code_size) {
            out_info_off = off;
            return true;
        }
        if (off < min_off + 4u) break;
    }
    return false;
}

static void calibrate_borealis_abi(agx_borealis_vk_ctx & vk) {
    // Tiny calibration shader with known push constants (16B) and local_size=(32,1,1)
    static const char * kCalibGlsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer OutBuf { uint data[]; };
layout(push_constant) uniform PC {
    uint64_t out_addr;
    uint magic;
    uint count;
} pc;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.count) {
        OutBuf(pc.out_addr).data[idx] = pc.magic + idx;
    }
}
)GLSL";

    agx_compile_opts opts{};
    opts.push_constant_bytes = 16u;
    std::vector<uint32_t> spv;
    std::string err;
    if (!run_glsl_to_spirv(kCalibGlsl, "", opts, spv, err)) {
        AGX_LOG_WARN("Borealis ABI calibration GLSL compile skipped: %s", err.c_str());
        return;
    }

    agx_shader_stats st{};
    extract_pipe_stats(vk, spv.data(), spv.size(), "main", 16u, st);

    std::vector<uint8_t> blob;
    if (!extract_shader_ext_blob(vk, spv.data(), spv.size(), "main", 16u, blob)) {
        AGX_LOG_WARN("Borealis ABI calibration vkGetShaderBinaryDataEXT failed");
        return;
    }

    size_t code_hdr_off = 0;
    uint32_t code_size = 0, data_size = 0;
    if (!locate_code_tail(blob, st.code_bytes, code_hdr_off, code_size, data_size)) {
        AGX_LOG_WARN("Borealis ABI calibration could not locate code tail");
        return;
    }

    size_t info_off = 0;
    if (!locate_agx_shader_info(blob, code_hdr_off, code_size, info_off)) {
        AGX_LOG_WARN("Borealis ABI calibration could not locate agx_shader_info");
        return;
    }

    // 1. Locate `agx3_uniform_dma` span matching our 16B push constant (dest=4, halfs=8)
    //    and `base_group` span (dest=12, halfs=6, offset_B=0x98).
    bool found_dma = false;
    for (size_t rel = 0x380; rel + 32 <= (code_hdr_off - info_off); rel += 4) {
        const uint8_t * p0 = blob.data() + info_off + rel;
        const uint8_t * p1 = p0 + 16;
        // Span 0: source=0, pointer=0, descriptor_B=0, dest=4, halfs=8, offset_B in [0x100..0x800]
        // Span 1: source=0, pointer=0, descriptor_B=0, dest=12, halfs=6, offset_B == 0x98
        if (p0[0] == 0 && rd_u16(p0 + 2) == 0 && rd_u16(p0 + 4) == 0 &&
            rd_u16(p0 + 6) == 4u && rd_u16(p0 + 8) == 8u &&
            p1[0] == 0 && rd_u16(p1 + 2) == 0 && rd_u16(p1 + 4) == 0 &&
            rd_u16(p1 + 6) == 12u && rd_u16(p1 + 8) == 6u &&
            rd_u32(p1 + 12) == 0x98u) {
            const uint32_t push_off = rd_u32(p0 + 12);
            vk.abi.dma_array_rel     = rel;
            vk.abi.dma_count_rel     = rel - 3u;
            vk.abi.push_offset_bytes = push_off;
            vk.abi.local_size_rel    = rel + 6u * 16u;
            vk.abi.scratch_size_rel  = vk.abi.local_size_rel + 4u;
            vk.abi.spectpr_rel       = vk.abi.scratch_size_rel + 8u;
            vk.abi.main_size_rel     = vk.abi.spectpr_rel + 4u;
            vk.abi.preamble_size_rel = vk.abi.main_size_rel + 4u;
            vk.abi.has_preamble_rel  = vk.abi.preamble_size_rel + 4u;
            found_dma = true;
            break;
        }
    }

    // 2. Locate `workgroup_size[3] == {32, 1, 1}`
    for (size_t rel = vk.abi.main_size_rel; rel + 6 <= (code_hdr_off - info_off); rel += 2) {
        const uint8_t * p = blob.data() + info_off + rel;
        if (rd_u16(p) == 32u && rd_u16(p + 2) == 1u && rd_u16(p + 4) == 1u) {
            vk.abi.wg_size_rel = rel;
            // `agx3_staging_known` sits 20 bytes before `workgroup_size`, `agx3_staging_halfs` 22 bytes before, `nr_gprs` 26 bytes before.
            if (rel >= 26u && blob[info_off + rel - 20u] == 1u) {
                vk.abi.nr_gprs_rel       = rel - 26u;
                vk.abi.staging_halfs_rel = rel - 22u;
                vk.abi.staging_known_rel = rel - 20u;
            }
            break;
        }
    }

    // 3. Locate `agx2_stats` (matching instructions + code_size from VK_KHR_pipeline_executable_properties)
    if (st.stats_valid && st.instructions > 0) {
        for (size_t rel = vk.abi.wg_size_rel; rel + 24 <= (code_hdr_off - info_off); rel += 4) {
            const uint8_t * p = blob.data() + info_off + rel;
            if (rd_u32(p) == st.instructions && rd_u32(p + 16) == code_size) {
                vk.abi.stats_rel  = rel;
                vk.abi.rodata_rel = rel - 8u;
                break;
            }
        }
    }

    vk.abi.calibrated = found_dma;
    AGX_LOG_INFO("Borealis ABI calibrated: push_offset=0x%x (%u B), dma_rel=0x%zx, wg_rel=0x%zx, stats_rel=0x%zx",
                 vk.abi.push_offset_bytes, vk.abi.push_offset_bytes,
                 vk.abi.dma_array_rel, vk.abi.wg_size_rel, vk.abi.stats_rel);
}

static bool init_borealis_vk(agx_borealis_vk_ctx & vk) {
    if (vk.initialized) return true;
    if (vk.failed)      return false;

    // Point Vulkan loader directly at Borealis Asahi ICD if not overridden
    if (!std::getenv("VK_DRIVER_FILES") && !std::getenv("VK_ICD_FILENAMES")) {
        const char * default_icds[] = {
            "/home/astaff/llama/mesa-retest-20261125/install/share/vulkan/icd.d/asahi_icd.aarch64.json",
            "/run/opengl-driver/share/vulkan/icd.d/asahi_icd.aarch64.json",
        };
        for (const char * icd_json : default_icds) {
            if (::access(icd_json, R_OK) == 0) {
                ::setenv("VK_DRIVER_FILES", icd_json, 0);
                break;
            }
        }
    }
    // Enable VK_KHR_cooperative_matrix on Honeykrisp/Borealis (requires HK_PERFTEST=...,coopmat)
    const char * cur_perf = std::getenv("HK_PERFTEST");
    if (!cur_perf || !cur_perf[0]) {
        ::setenv("HK_PERFTEST", "weakbarrier,coopmat", 0);
    } else if (std::strstr(cur_perf, "coopmat") == nullptr) {
        std::string merged = std::string(cur_perf) + ",coopmat";
        ::setenv("HK_PERFTEST", merged.c_str(), 1);
    }

    // Disable USC preambles by default so Borealis uses uniform DMA + root pointer cleanly
    const char * cur_dbg = std::getenv("AGX_MESA_DEBUG");
    if (!cur_dbg || !cur_dbg[0]) {
        ::setenv("AGX_MESA_DEBUG", "nopreamble", 0);
    } else if (std::strstr(cur_dbg, "nopreamble") == nullptr) {
        std::string merged = std::string(cur_dbg) + ",nopreamble";
        ::setenv("AGX_MESA_DEBUG", merged.c_str(), 1);
    }

    std::vector<const char *> vk_candidates;
    if (const char * env_icd = std::getenv("AGX_VULKAN_ICD_LIB"); env_icd && env_icd[0]) {
        vk_candidates.push_back(env_icd);
    }
    vk_candidates.push_back("/home/astaff/llama/mesa-retest-20261125/install/lib/libvulkan_asahi.so");
    vk_candidates.push_back("libvulkan.so.1");
    vk_candidates.push_back("/run/current-system/sw/lib/libvulkan.so.1");
    vk_candidates.push_back("/run/opengl-driver/lib/libvulkan_asahi.so");

    const char * loaded_vk_lib = nullptr;
    for (const char * lib : vk_candidates) {
        vk.dl_handle = ::dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (vk.dl_handle) {
            loaded_vk_lib = lib;
            break;
        }
    }
    if (!vk.dl_handle) {
        AGX_LOG_ERROR("Failed to dlopen Borealis/Vulkan ICD library: %s", ::dlerror());
        vk.failed = true;
        return false;
    }

    auto pGetInstProc = reinterpret_cast<PFN_agx_vkGetInstanceProcAddr>(::dlsym(vk.dl_handle, "vkGetInstanceProcAddr"));
    if (!pGetInstProc) {
        pGetInstProc = reinterpret_cast<PFN_agx_vkGetInstanceProcAddr>(::dlsym(vk.dl_handle, "vk_icdGetInstanceProcAddr"));
    }
    if (!pGetInstProc) {
        AGX_LOG_ERROR("vkGetInstanceProcAddr / vk_icdGetInstanceProcAddr missing in Vulkan library");
        vk.failed = true;
        return false;
    }

    auto pCreateInstance = reinterpret_cast<PFN_agx_vkCreateInstance>(pGetInstProc(nullptr, "vkCreateInstance"));
    if (!pCreateInstance) {
        vk.failed = true;
        return false;
    }

    AgxVkApplicationInfo app{};
    app.sType            = AGX_VK_STYPE_APPLICATION_INFO;
    app.pApplicationName = "ggml-agx-compiler";
    app.apiVersion       = AGX_VK_API_VERSION_1_3;

    AgxVkInstanceCreateInfo ici{};
    ici.sType            = AGX_VK_STYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    if (pCreateInstance(&ici, nullptr, &vk.instance) != AGX_VK_SUCCESS || !vk.instance) {
        AGX_LOG_ERROR("vkCreateInstance failed for Borealis compiler service");
        vk.failed = true;
        return false;
    }

    vk.destroy_instance = reinterpret_cast<PFN_agx_vkDestroyInstance>(pGetInstProc(vk.instance, "vkDestroyInstance"));
    auto pEnumDevs      = reinterpret_cast<PFN_agx_vkEnumeratePhysicalDevices>(pGetInstProc(vk.instance, "vkEnumeratePhysicalDevices"));
    auto pEnumExts      = reinterpret_cast<PFN_agx_vkEnumerateDeviceExtensionProperties>(pGetInstProc(vk.instance, "vkEnumerateDeviceExtensionProperties"));
    auto pGetFeats2     = reinterpret_cast<PFN_agx_vkGetPhysicalDeviceFeatures2>(pGetInstProc(vk.instance, "vkGetPhysicalDeviceFeatures2"));
    auto pCreateDevice  = reinterpret_cast<PFN_agx_vkCreateDevice>(pGetInstProc(vk.instance, "vkCreateDevice"));
    auto pGetDevProc    = reinterpret_cast<PFN_agx_vkGetDeviceProcAddr>(pGetInstProc(vk.instance, "vkGetDeviceProcAddr"));

    uint32_t ndev = 0;
    pEnumDevs(vk.instance, &ndev, nullptr);
    if (ndev == 0) {
        AGX_LOG_ERROR("No Vulkan physical device found for Borealis compiler service");
        vk.failed = true;
        return false;
    }
    std::vector<AgxVkPhysicalDevice> pdevs(ndev);
    pEnumDevs(vk.instance, &ndev, pdevs.data());
    vk.phys_dev = pdevs[0];

    // Query supported device extensions
    uint32_t n_ext = 0;
    pEnumExts(vk.phys_dev, nullptr, &n_ext, nullptr);
    std::vector<AgxVkExtensionProperties> avail_exts(n_ext);
    if (n_ext > 0) {
        pEnumExts(vk.phys_dev, nullptr, &n_ext, avail_exts.data());
    }
    auto has_ext = [&](const char * name) -> bool {
        for (const auto & e : avail_exts) {
            if (std::strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };

    std::vector<const char *> enabled_exts;
    const char * desired_exts[] = {
        "VK_EXT_shader_object",
        "VK_KHR_pipeline_executable_properties",
        "VK_KHR_buffer_device_address",
        "VK_KHR_cooperative_matrix",
        "VK_KHR_shader_bfloat16",
        "VK_KHR_shader_float_controls2",
        "VK_KHR_shader_subgroup_rotate",
    };
    for (const char * ext : desired_exts) {
        if (has_ext(ext)) {
            enabled_exts.push_back(ext);
        }
    }

    // Query physical device features cleanly via pNext chain so we only enable supported bits
    AgxVkPhysicalDeviceShaderObjectFeaturesEXT so_feat{};
    so_feat.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;

    AgxVkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_feat{};
    pep_feat.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEAT;
    pep_feat.pNext = &so_feat;

    AgxVkPhysicalDeviceCooperativeMatrixFeaturesKHR cm_feat{};
    cm_feat.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;

    AgxVkPhysicalDeviceShaderBfloat16FeaturesKHR bf16_feat{};
    bf16_feat.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;

    void * chain_tail = &pep_feat;
    if (has_ext("VK_KHR_cooperative_matrix")) {
        cm_feat.pNext = chain_tail;
        chain_tail = &cm_feat;
    }
    if (has_ext("VK_KHR_shader_bfloat16")) {
        bf16_feat.pNext = chain_tail;
        chain_tail = &bf16_feat;
    }

    AgxVkPhysicalDeviceVulkan13Features v13{};
    v13.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.pNext = chain_tail;

    AgxVkPhysicalDeviceVulkan12Features v12{};
    v12.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.pNext = &v13;

    AgxVkPhysicalDeviceVulkan11Features v11{};
    v11.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    v11.pNext = &v12;

    AgxVkPhysicalDeviceFeatures2 f2{};
    f2.sType = AGX_VK_STYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &v11;

    pGetFeats2(vk.phys_dev, &f2);
    vk.has_coopmat = has_ext("VK_KHR_cooperative_matrix") && (cm_feat.cooperativeMatrix == AGX_VK_TRUE);

    float prio = 1.0f;
    AgxVkDeviceQueueCreateInfo qci{};
    qci.sType            = AGX_VK_STYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    AgxVkDeviceCreateInfo dci{};
    dci.sType                   = AGX_VK_STYPE_DEVICE_CREATE_INFO;
    dci.pNext                   = &f2;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = static_cast<uint32_t>(enabled_exts.size());
    dci.ppEnabledExtensionNames = enabled_exts.data();

    if (pCreateDevice(vk.phys_dev, &dci, nullptr, &vk.device) != AGX_VK_SUCCESS || !vk.device) {
        AGX_LOG_ERROR("vkCreateDevice failed for Borealis compiler service");
        vk.failed = true;
        return false;
    }

    vk.destroy_device          = reinterpret_cast<PFN_agx_vkDestroyDevice>(pGetDevProc(vk.device, "vkDestroyDevice"));
    vk.create_shaders_ext      = reinterpret_cast<PFN_agx_vkCreateShadersEXT>(pGetDevProc(vk.device, "vkCreateShadersEXT"));
    vk.get_shader_bin_ext      = reinterpret_cast<PFN_agx_vkGetShaderBinaryDataEXT>(pGetDevProc(vk.device, "vkGetShaderBinaryDataEXT"));
    vk.destroy_shader_ext      = reinterpret_cast<PFN_agx_vkDestroyShaderEXT>(pGetDevProc(vk.device, "vkDestroyShaderEXT"));
    vk.create_shader_module    = reinterpret_cast<PFN_agx_vkCreateShaderModule>(pGetDevProc(vk.device, "vkCreateShaderModule"));
    vk.destroy_shader_module   = reinterpret_cast<PFN_agx_vkDestroyShaderModule>(pGetDevProc(vk.device, "vkDestroyShaderModule"));
    vk.create_pipeline_layout  = reinterpret_cast<PFN_agx_vkCreatePipelineLayout>(pGetDevProc(vk.device, "vkCreatePipelineLayout"));
    vk.destroy_pipeline_layout = reinterpret_cast<PFN_agx_vkDestroyPipelineLayout>(pGetDevProc(vk.device, "vkDestroyPipelineLayout"));
    vk.create_compute_pipes    = reinterpret_cast<PFN_agx_vkCreateComputePipelines>(pGetDevProc(vk.device, "vkCreateComputePipelines"));
    vk.destroy_pipeline        = reinterpret_cast<PFN_agx_vkDestroyPipeline>(pGetDevProc(vk.device, "vkDestroyPipeline"));
    vk.get_pipe_exec_props     = reinterpret_cast<PFN_agx_vkGetPipelineExecutablePropertiesKHR>(pGetDevProc(vk.device, "vkGetPipelineExecutablePropertiesKHR"));
    vk.get_pipe_exec_stats     = reinterpret_cast<PFN_agx_vkGetPipelineExecutableStatisticsKHR>(pGetDevProc(vk.device, "vkGetPipelineExecutableStatisticsKHR"));

    vk.initialized = true;
    AGX_LOG_INFO("Borealis compiler service loaded from %s (coopmat=%d)",
                 loaded_vk_lib ? loaded_vk_lib : "<unknown>", (int) vk.has_coopmat);
    calibrate_borealis_abi(vk);
    return true;
}

static void shutdown_borealis_vk(agx_borealis_vk_ctx & vk) {
    if (vk.device && vk.destroy_device) {
        vk.destroy_device(vk.device, nullptr);
        vk.device = nullptr;
    }
    if (vk.instance && vk.destroy_instance) {
        vk.destroy_instance(vk.instance, nullptr);
        vk.instance = nullptr;
    }
    if (vk.dl_handle) {
        ::dlclose(vk.dl_handle);
        vk.dl_handle = nullptr;
    }
    vk.initialized = false;
}

// ============================================================================
// Disk Cache Serialization (`.agxbin`)
// ============================================================================

struct agxbin_header {
    uint32_t magic;
    uint32_t version;
    uint32_t main_offset;
    uint32_t main_size;
    uint32_t binary_size;
    uint32_t const_data_size;
    uint32_t local_size[3];
    uint32_t threadgroup_bytes;
    uint32_t scratch_bytes;
    uint32_t staging_halfs;
    uint16_t root_uniform_half;
    uint16_t uses_root;
    uint32_t push_offset_bytes;
    uint32_t root_size_bytes;
    uint16_t rodata_base_half;
    uint16_t rodata_size_16;
    uint32_t rodata_offset;
    uint32_t uniform_dma_count;
    agx_shader_stats stats;
};

constexpr uint32_t AGXBIN_MAGIC   = 0x33584741u; // "AGX3"
constexpr uint32_t AGXBIN_VERSION = 3u;

static bool save_agxbin(const std::string & path,
                        const agx_kernel & k,
                        const void * code_ptr, size_t code_sz,
                        const void * data_ptr, size_t data_sz) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;

    agxbin_header hdr{};
    hdr.magic             = AGXBIN_MAGIC;
    hdr.version           = AGXBIN_VERSION;
    hdr.main_offset       = k.main_offset;
    hdr.main_size         = k.main_size;
    hdr.binary_size       = static_cast<uint32_t>(code_sz);
    hdr.const_data_size   = static_cast<uint32_t>(data_sz);
    hdr.local_size[0]     = k.local_size[0];
    hdr.local_size[1]     = k.local_size[1];
    hdr.local_size[2]     = k.local_size[2];
    hdr.threadgroup_bytes = k.threadgroup_bytes;
    hdr.scratch_bytes     = k.scratch_bytes;
    hdr.staging_halfs     = k.staging_halfs;
    hdr.root_uniform_half = k.root_uniform_half;
    hdr.uses_root         = k.uses_root ? 1u : 0u;
    hdr.push_offset_bytes = k.push_offset_bytes;
    hdr.root_size_bytes   = k.root_size_bytes;
    hdr.rodata_base_half  = k.rodata_base_half;
    hdr.rodata_size_16    = k.rodata_size_16;
    hdr.rodata_offset     = k.rodata_offset;
    hdr.uniform_dma_count = static_cast<uint32_t>(k.uniform_dma.size());
    hdr.stats             = k.stats;

    ofs.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    if (!k.uniform_dma.empty()) {
        ofs.write(reinterpret_cast<const char *>(k.uniform_dma.data()),
                  static_cast<std::streamsize>(k.uniform_dma.size() * sizeof(agx_uniform_dma_span)));
    }
    if (code_sz > 0) {
        ofs.write(reinterpret_cast<const char *>(code_ptr), static_cast<std::streamsize>(code_sz));
    }
    if (data_sz > 0) {
        ofs.write(reinterpret_cast<const char *>(data_ptr), static_cast<std::streamsize>(data_sz));
    }
    return ofs.good();
}

static bool load_agxbin(const std::string & path,
                        agx_kernel & k,
                        std::vector<uint8_t> & out_code,
                        std::vector<uint8_t> & out_data) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    agxbin_header hdr{};
    ifs.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!ifs || hdr.magic != AGXBIN_MAGIC || hdr.version != AGXBIN_VERSION || hdr.binary_size == 0) {
        return false;
    }

    k.main_offset       = hdr.main_offset;
    k.main_size         = hdr.main_size;
    k.binary_size       = hdr.binary_size;
    k.local_size[0]     = hdr.local_size[0];
    k.local_size[1]     = hdr.local_size[1];
    k.local_size[2]     = hdr.local_size[2];
    k.threadgroup_bytes = hdr.threadgroup_bytes;
    k.scratch_bytes     = hdr.scratch_bytes;
    k.staging_halfs     = hdr.staging_halfs;
    k.root_uniform_half = hdr.root_uniform_half;
    k.uses_root         = hdr.uses_root != 0;
    k.push_offset_bytes = hdr.push_offset_bytes;
    k.root_size_bytes   = hdr.root_size_bytes;
    k.rodata_base_half  = hdr.rodata_base_half;
    k.rodata_size_16    = hdr.rodata_size_16;
    k.rodata_offset     = hdr.rodata_offset;
    k.stats             = hdr.stats;

    k.uniform_dma.resize(hdr.uniform_dma_count);
    if (hdr.uniform_dma_count > 0) {
        ifs.read(reinterpret_cast<char *>(k.uniform_dma.data()),
                 static_cast<std::streamsize>(hdr.uniform_dma_count * sizeof(agx_uniform_dma_span)));
    }
    out_code.resize(hdr.binary_size);
    ifs.read(reinterpret_cast<char *>(out_code.data()), static_cast<std::streamsize>(hdr.binary_size));
    out_data.resize(hdr.const_data_size);
    if (hdr.const_data_size > 0) {
        ifs.read(reinterpret_cast<char *>(out_data.data()), static_cast<std::streamsize>(hdr.const_data_size));
    }
    return ifs.good();
}

// Build Apple's G17P/HAL200 +0x3c0 AGX3 code container around `code_ptr[0..code_sz)`.
// LDSHDR names entry = container_base + 0x3c0 + main_offset; the shader loader reads
// block_size at entry - 0x80 and the minimal constant program at entry - 0x40.
// (Mirrors Mesa src/asahi/lib/agx_linker.c & agx3_code_container_core.c.)
static constexpr size_t AGX3_CONTAINER_LEAD_BYTES       = 0x340u;
static constexpr size_t AGX3_CONTAINER_BLOCK_HDR_BYTES  = 0x40u;
static constexpr size_t AGX3_CONTAINER_CONST_PROG_BYTES = 0x40u;
static constexpr size_t AGX3_CONTAINER_ENTRY_OFFSET     = 0x3c0u;

static void build_g17p_code_container(uint8_t * dst,
                                      uint32_t cores_per_cluster,
                                      const void * code_ptr,
                                      size_t code_sz) {
    static constexpr uint8_t kHelperTerminal[16] = {
        0xf7, 0x03, 0xaa, 0x00, 0x8f, 0x02, 0x54, 0x01,
        0x06, 0x00, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
    };
    static constexpr uint8_t kComputeConstProg[14] = {
        0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00,
        0x00, 0x60, 0x00, 0x0e, 0x00, 0x00, 0x00,
    };

    const uint32_t lead = static_cast<uint32_t>(AGX3_CONTAINER_LEAD_BYTES);
    std::memcpy(dst, &lead, sizeof(lead));
    for (size_t off = 0x40u; off < AGX3_CONTAINER_LEAD_BYTES; off += 2u) {
        dst[off]     = 0x06;
        dst[off + 1] = 0x00;
    }

    const uint32_t slots = std::min<uint32_t>(14u, std::max<uint32_t>(6u, cores_per_cluster));
    auto fill_table = [&](size_t base) {
        for (uint32_t i = 0; i < slots; ++i) {
            uint8_t * rec = dst + base + i * 0x10u;
            std::memset(rec, 0, 0x10u);
            rec[0] = 0x0f;
            rec[2] = 0x54;
            rec[3] = static_cast<uint8_t>((slots - i) * 0x10u);
            for (size_t h = 10; h < 16; h += 2) {
                rec[h]     = 0x06;
                rec[h + 1] = 0x00;
            }
        }
        std::memcpy(dst + base + slots * 0x10u, kHelperTerminal, 16u);
        std::memcpy(dst + base + (slots + 1u) * 0x10u, kHelperTerminal, 16u);
    };
    fill_table(0x100u);
    fill_table(0x200u);

    const uint32_t block_size = static_cast<uint32_t>(
        AGX3_CONTAINER_BLOCK_HDR_BYTES + AGX3_CONTAINER_CONST_PROG_BYTES + agx_align_up(code_sz, 64u));
    std::memcpy(dst + AGX3_CONTAINER_LEAD_BYTES, &block_size, sizeof(block_size));

    uint8_t * cprog = dst + AGX3_CONTAINER_LEAD_BYTES + AGX3_CONTAINER_BLOCK_HDR_BYTES;
    for (size_t off = sizeof(kComputeConstProg); off < AGX3_CONTAINER_CONST_PROG_BYTES; off += 2u) {
        cprog[off]     = 0x06;
        cprog[off + 1] = 0x00;
    }
    std::memcpy(cprog, kComputeConstProg, sizeof(kComputeConstProg));

    std::memcpy(dst + AGX3_CONTAINER_ENTRY_OFFSET, code_ptr, code_sz);
}

static bool spirv_uses_coopmat(const uint32_t * words, size_t count) {
    if (!words || count < 5 || words[0] != 0x07230203u) return false;
    size_t i = 5;
    while (i < count) {
        const uint32_t hdr = words[i];
        const uint32_t wc  = hdr >> 16;
        const uint32_t op  = hdr & 0xffffu;
        if (wc == 0 || i + wc > count) break;
        // OpCapability = 17, CapabilityCooperativeMatrixKHR = 6022
        if (op == 17u && wc >= 2u && words[i + 1] == 6022u) {
            return true;
        }
        i += wc;
    }
    return false;
}

static bool upload_kernel_to_usc(agx_drm_device * dev,
                                 agx_kernel & k,
                                 const void * code_ptr,
                                 size_t code_sz,
                                 const void * data_ptr,
                                 size_t data_sz) {
    if (!dev || !code_ptr || code_sz == 0) {
        return false;
    }
    const agx_drm_device_info & dinfo = agx_drm_get_info(dev);
    const auto * raw_code = static_cast<const uint8_t *>(code_ptr);

    // Resolve the executable `main` slice inside `code_ptr[0..code_sz)`.
    // Borealis places `rodata` at offset 0 (padded to AGX_CODE_ALIGN = 128 B) ahead of `main`,
    // whereas Apple's G17P +0x3c0 container requires `main` immediately at `shader_va = base + 0x3c0`
    // so the USC loader finds `block_size` at `shader_va - 0x80` and `kComputeConstProg` at `shader_va - 0x40`.
    const uint32_t main_off = k.main_offset;
    const size_t   main_sz  = (k.main_size > 0u) ? static_cast<size_t>(k.main_size) : (code_sz - main_off);
    if (main_off > code_sz || main_sz == 0 || static_cast<size_t>(main_off) + main_sz > code_sz) {
        AGX_LOG_ERROR("Invalid main slice [0x%x + %zu] within %zu B binary for kernel '%s'",
                      main_off, main_sz, code_sz, k.name.c_str());
        return false;
    }

    const uint32_t orig_ro_off   = k.rodata_offset;
    const size_t   ro_raw_bytes  = static_cast<size_t>(k.rodata_size_16) * 2u;
    const size_t   ro_copy_align = agx_align_up(ro_raw_bytes, 4u);
    if (k.rodata_size_16 != 0 && static_cast<size_t>(orig_ro_off) + ro_raw_bytes > code_sz) {
        AGX_LOG_ERROR("Invalid rodata slice [0x%x + %zu] within %zu B binary for kernel '%s'",
                      orig_ro_off, ro_raw_bytes, code_sz, k.name.c_str());
        return false;
    }

    // Allocate in the mandatory bit-40 USC executable VA window (>= 0x10000000000):
    //   [0x000 .. 0x340): G17P container lead + helper tables
    //   [0x340 .. 0x380): block header (block_size)
    //   [0x380 .. 0x3c0): minimal compute constant program
    //   [0x3c0 .. +main_sz): executable AGX3B `main` machine code (`shader_va = base + 0x3c0`)
    //   [+256B prefetch pad]
    //   [ro_dst_off ..): 64B-aligned rodata constants (read by Ioto LDDEV)
    const size_t container_end = AGX3_CONTAINER_ENTRY_OFFSET + agx_align_up(main_sz, 64u) + 256u;
    const size_t ro_dst_off    = agx_align_up(container_end, 64u);
    const size_t total_sz      = (k.rodata_size_16 != 0)
                                     ? (ro_dst_off + agx_align_up(ro_copy_align, 64u))
                                     : container_end;
    const size_t alloc_sz      = agx_align_up(total_sz, 128u);

    std::string lbl = "usc:" + k.name;
    k.code_bo = agx_bo_alloc(dev, alloc_sz, AGX_BO_EXEC | AGX_BO_ZERO, lbl.c_str());
    if (!k.code_bo) {
        AGX_LOG_ERROR("Failed to allocate USC executable BO (%zu B) for kernel '%s'", alloc_sz, k.name.c_str());
        return false;
    }

    auto * bo_dst = static_cast<uint8_t *>(k.code_bo->cpu_ptr);
    build_g17p_code_container(bo_dst, dinfo.cores_per_cluster, raw_code + main_off, main_sz);
    k.shader_va   = k.code_bo->gpu_va + AGX3_CONTAINER_ENTRY_OFFSET;
    k.binary_size = static_cast<uint32_t>(code_sz);
    k.main_size   = static_cast<uint32_t>(main_sz);
    if (k.rodata_size_16 != 0) {
        std::memcpy(bo_dst + ro_dst_off, raw_code + orig_ro_off, ro_raw_bytes);
        k.rodata_offset = static_cast<uint32_t>(ro_dst_off);
    }

    if (data_ptr && data_sz > 0) {
        std::string dlbl = "rodata:" + k.name;
        k.const_data_bo = agx_bo_alloc(dev, agx_align_up(data_sz, 128u),
                                       AGX_BO_SLC_PIN | AGX_BO_READONLY | AGX_BO_ZERO, dlbl.c_str());
        if (!k.const_data_bo) {
            agx_bo_free(dev, k.code_bo);
            k.code_bo = nullptr;
            return false;
        }
        std::memcpy(k.const_data_bo->cpu_ptr, data_ptr, data_sz);
        k.const_data_va = k.const_data_bo->gpu_va;
    }
    return true;
}

} // namespace

// ============================================================================
// Public Compiler Implementation
// ============================================================================

struct agx_compiler {
    agx_drm_device * dev = nullptr;
    std::string      kernel_dir;
    std::string      cache_dir;
    bool             use_cache = true;

    std::mutex                                              mtx;
    agx_borealis_vk_ctx                                     vk;
    std::unordered_map<std::string, std::unique_ptr<agx_kernel>> kernels;
    std::vector<std::string>                                order;
};

agx_compiler * agx_compiler_create(agx_drm_device * dev,
                                   const char *     kernel_dir,
                                   const char *     cache_dir) {
    if (!dev) {
        return nullptr;
    }
    auto * c = new agx_compiler();
    c->dev = dev;

    if (kernel_dir && kernel_dir[0]) {
        c->kernel_dir = kernel_dir;
    } else if (const char * env = std::getenv("GGML_AGX_KERNEL_DIR"); env && env[0]) {
        c->kernel_dir = env;
    }

    if (std::getenv("GGML_AGX_NO_CACHE") != nullptr) {
        c->use_cache = false;
    } else if (cache_dir && cache_dir[0]) {
        c->cache_dir = cache_dir;
    } else if (const char * env = std::getenv("GGML_AGX_CACHE_DIR"); env && env[0]) {
        c->cache_dir = env;
    } else if (const char * home = std::getenv("HOME"); home && home[0]) {
        c->cache_dir = std::string(home) + "/.cache/ggml-agx";
    } else {
        c->cache_dir = "/tmp/ggml-agx-cache";
    }

    if (c->use_cache && !c->cache_dir.empty()) {
        std::error_code ec;
        fs::create_directories(c->cache_dir, ec);
        if (ec) {
            c->use_cache = false;
        }
    }
    return c;
}

void agx_compiler_destroy(agx_compiler * c) {
    if (!c) return;
    {
        std::lock_guard<std::mutex> lock(c->mtx);
        for (auto & kv : c->kernels) {
            if (kv.second) {
                if (kv.second->code_bo)       agx_bo_free(c->dev, kv.second->code_bo);
                if (kv.second->const_data_bo) agx_bo_free(c->dev, kv.second->const_data_bo);
            }
        }
        c->kernels.clear();
        shutdown_borealis_vk(c->vk);
    }
    delete c;
}

const agx_kernel * agx_compiler_get(const agx_compiler * c, const char * name) {
    if (!c || !name) return nullptr;
    auto * mc = const_cast<agx_compiler *>(c);
    std::lock_guard<std::mutex> lock(mc->mtx);
    auto it = mc->kernels.find(name);
    return (it != mc->kernels.end()) ? it->second.get() : nullptr;
}

const agx_kernel * agx_compiler_compile_spirv(agx_compiler *           c,
                                              const char *             name,
                                              const uint32_t *         spirv_words,
                                              size_t                   word_count,
                                              const agx_compile_opts & opts) {
    if (!c || !name || !spirv_words || word_count < 5) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    if (auto it = c->kernels.find(name); it != c->kernels.end()) {
        return it->second.get();
    }

    auto k = std::make_unique<agx_kernel>();
    k->name          = name;
    k->is_hot_kernel = opts.is_hot_kernel;
    parse_spirv_local_size(spirv_words, word_count, k->local_size);

    const agx_drm_device_info & dinfo = agx_drm_get_info(c->dev);
    const std::string cache_path = c->use_cache
        ? (c->cache_dir + "/" + compute_cache_key(spirv_words, word_count, opts, dinfo) + ".agxbin")
        : std::string();

    std::vector<uint8_t> code_bytes;
    std::vector<uint8_t> const_bytes;
    bool loaded_from_cache = false;

    if (!cache_path.empty() && load_agxbin(cache_path, *k, code_bytes, const_bytes)) {
        loaded_from_cache = true;
    }

    if (!loaded_from_cache) {
        if (!init_borealis_vk(c->vk)) {
            AGX_LOG_ERROR("Cannot compile kernel '%s': Borealis compiler service unavailable", name);
            return nullptr;
        }
        if (spirv_uses_coopmat(spirv_words, word_count) && !c->vk.has_coopmat) {
            AGX_LOG_ERROR("Cannot compile kernel '%s': SPIR-V requires CooperativeMatrixKHR, not enabled on Borealis", name);
            return nullptr;
        }

        const char * ep = opts.entry_point ? opts.entry_point : "main";
        const uint32_t push_B = std::min<uint32_t>(opts.push_constant_bytes, AGX_MAX_PUSH_BYTES);

        // 1. Query official shaderdb statistics via VK_KHR_pipeline_executable_properties
        extract_pipe_stats(c->vk, spirv_words, word_count, ep, push_B, k->stats);

        // 2. Extract native AGX3B binary + metadata via VK_EXT_shader_object
        std::vector<uint8_t> blob;
        if (!extract_shader_ext_blob(c->vk, spirv_words, word_count, ep, push_B, blob)) {
            AGX_LOG_ERROR("vkGetShaderBinaryDataEXT failed for kernel '%s'", name);
            return nullptr;
        }

        size_t code_hdr_off = 0;
        uint32_t code_size = 0, data_size = 0;
        if (!locate_code_tail(blob, k->stats.code_bytes, code_hdr_off, code_size, data_size)) {
            AGX_LOG_ERROR("Failed to parse Borealis binary tail for kernel '%s' (blob=%zu)", name, blob.size());
            return nullptr;
        }

        code_bytes.assign(blob.data() + code_hdr_off + 8u,
                          blob.data() + code_hdr_off + 8u + code_size);
        if (data_size > 0) {
            const_bytes.assign(blob.data() + code_hdr_off + 8u + code_size,
                               blob.data() + code_hdr_off + 8u + code_size + data_size);
        }

        // Defaults before parsing agx_shader_info
        k->push_offset_bytes = c->vk.abi.push_offset_bytes;
        k->root_size_bytes   = static_cast<uint32_t>(agx_align_up(k->push_offset_bytes + push_B, 16u));
        k->uses_root         = true;
        k->root_uniform_half = 0u;
        k->main_offset       = 0u;
        k->main_size         = code_size;
        k->binary_size       = code_size;
        k->scratch_bytes     = k->stats.scratch_bytes;

        size_t info_off = 0;
        if (locate_agx_shader_info(blob, code_hdr_off, code_size, info_off)) {
            const uint8_t * info = blob.data() + info_off;
            const size_t info_avail = code_hdr_off - info_off;

            // `root_descriptor_size` sits immediately before `agx_shader_info` on HAL200!
            if (info_off >= 4u) {
                const uint32_t rds = rd_u32(blob.data() + info_off - 4u);
                if (rds >= 16u && rds <= 16384u) {
                    k->root_size_bytes = std::max(k->root_size_bytes, rds);
                }
            }

            // Parse `agx3_uniform_dma` spans (root/push constant & base_group uniform promotion)
            if (c->vk.abi.dma_array_rel + 6u * 16u <= info_avail) {
                const uint8_t dma_cnt = info[c->vk.abi.dma_count_rel];
                if (dma_cnt <= 6u) {
                    for (uint8_t i = 0; i < dma_cnt; ++i) {
                        const uint8_t * sp = info + c->vk.abi.dma_array_rel + i * 16u;
                        const uint8_t  source   = sp[0];
                        const uint16_t dest     = rd_u16(sp + 6u);
                        const uint16_t halfs    = rd_u16(sp + 8u);
                        const uint32_t offset_B = rd_u32(sp + 12u);
                        if (source == 0u && halfs > 0u && halfs <= 256u) {
                            agx_uniform_dma_span span{};
                            span.dest_half = dest;
                            span.halfs     = halfs;
                            span.offset_B  = offset_B;
                            span.set       = -1;
                            span.binding   = 0;
                            k->uniform_dma.push_back(span);
                        }
                    }
                }
            }

            // Parse threadgroup SRAM (`local_size`), `scratch_size`, `agx3_spectpr_scratch_size`, `main_size`
            if (c->vk.abi.main_size_rel + 4u <= info_avail) {
                const uint16_t tg_bytes  = rd_u16(info + c->vk.abi.local_size_rel);
                const uint32_t scr_bytes = rd_u32(info + c->vk.abi.scratch_size_rel);
                const uint32_t spectpr   = rd_u32(info + c->vk.abi.spectpr_rel);
                const uint32_t main_sz   = rd_u32(info + c->vk.abi.main_size_rel);
                k->threadgroup_bytes     = tg_bytes;
                k->stats.threadgroup_bytes = tg_bytes;
                k->scratch_bytes         = std::max(scr_bytes, spectpr);
                if (main_sz > 0u && main_sz <= code_size) {
                    k->main_size = main_sz;
                }
            }

            // Parse `nr_gprs` & `agx3_staging_halfs`
            if (c->vk.abi.staging_known_rel + 1u <= info_avail) {
                const uint16_t gprs16 = rd_u16(info + c->vk.abi.nr_gprs_rel);
                const uint16_t stg    = rd_u16(info + c->vk.abi.staging_halfs_rel);
                const uint8_t  known  = info[c->vk.abi.staging_known_rel];
                if (known == 1u) {
                    k->staging_halfs = stg;
                    k->stats.staging_halfs = stg;
                }
                if (k->stats.nr_gprs_16 == 0 && gprs16 > 0 && gprs16 <= 256u) {
                    k->stats.nr_gprs_16 = gprs16;
                    k->stats.nr_gprs_32 = (gprs16 + 1u) / 2u;
                }
            }

            // Parse `workgroup_size[3]`
            if (c->vk.abi.wg_size_rel + 6u <= info_avail) {
                const uint16_t lx = rd_u16(info + c->vk.abi.wg_size_rel);
                const uint16_t ly = rd_u16(info + c->vk.abi.wg_size_rel + 2u);
                const uint16_t lz = rd_u16(info + c->vk.abi.wg_size_rel + 4u);
                if (lx > 0 && ly > 0 && lz > 0 && static_cast<uint32_t>(lx) * ly * lz <= 1024u) {
                    k->local_size[0] = lx;
                    k->local_size[1] = ly;
                    k->local_size[2] = lz;
                }
            }

            // Parse compiler `agx_rodata` (constant LUTs placed before `main` in Borealis binary)
            if (c->vk.abi.rodata_rel + 8u <= info_avail) {
                const uint32_t ro_off  = rd_u32(info + c->vk.abi.rodata_rel);
                const uint16_t ro_base = rd_u16(info + c->vk.abi.rodata_rel + 4u);
                const uint16_t ro_sz16 = rd_u16(info + c->vk.abi.rodata_rel + 6u);
                if (ro_sz16 > 0 && ro_off + static_cast<uint32_t>(ro_sz16) * 2u <= code_size) {
                    k->rodata_offset    = ro_off;
                    k->rodata_base_half = ro_base;
                    k->rodata_size_16   = ro_sz16;
                }
            }

            // Compute exact `main_offset` inside `code_bytes`:
            // Borealis (`agx_compile.c:15333-15341, 19356-19360`) emits `rodata` at offset 0 padded
            // to AGX_CODE_ALIGN (128 B), followed by `main` of length `main_size = binary_size - main_offset`
            // (compute shaders have no UVS relocations and `AGX_MESA_DEBUG=nopreamble` disables preambles).
            const bool has_preamble =
                (c->vk.abi.has_preamble_rel < info_avail) && (info[c->vk.abi.has_preamble_rel] != 0u);
            k->has_preamble = has_preamble;
            if (!has_preamble && k->main_size > 0u && k->main_size < code_size) {
                const uint32_t derived_off = code_size - k->main_size;
                const uint32_t min_ro_off  = (k->rodata_size_16 > 0u)
                                                 ? static_cast<uint32_t>(agx_align_up(
                                                       k->rodata_offset + static_cast<uint32_t>(k->rodata_size_16) * 2u,
                                                       64u))
                                                 : 0u;
                if ((derived_off & 0x3fu) == 0u && derived_off >= min_ro_off) {
                    k->main_offset = derived_off;
                }
            } else if (k->rodata_size_16 > 0u && k->rodata_offset == 0u && k->main_offset == 0u) {
                const uint32_t ro_end = static_cast<uint32_t>(
                    agx_align_up(static_cast<uint32_t>(k->rodata_size_16) * 2u, 128u));
                if (ro_end < code_size) {
                    k->main_offset = ro_end;
                    k->main_size   = code_size - ro_end;
                }
            }
        }

        if (k->stats.code_bytes == 0) {
            k->stats.code_bytes = code_size;
        }
        if (k->stats.nr_gprs_32 == 0 && k->stats.nr_gprs_16 > 0) {
            k->stats.nr_gprs_32 = (k->stats.nr_gprs_16 + 1u) / 2u;
        }

        if (!cache_path.empty()) {
            save_agxbin(cache_path, *k, code_bytes.data(), code_bytes.size(),
                        const_bytes.data(), const_bytes.size());
        }
    }

    // Pillar 1 & 2 strict shaderdb enforcement (`0:0` spills, `<= 96` 32-bit GPRs on hot kernels)
    const bool allow_spills = (std::getenv("GGML_AGX_ALLOW_SPILLS") != nullptr);
    if (k->stats.stats_valid) {
        if ((k->stats.spills > 0 || k->stats.fills > 0) && opts.strict_zero_spill && !allow_spills) {
            AGX_LOG_ERROR("Kernel '%s' FAILED strict 0:0 spill check (%u:%u spills:fills, %u GPR32, %u B scratch)",
                          name, k->stats.spills, k->stats.fills, k->stats.nr_gprs_32, k->stats.scratch_bytes);
            return nullptr;
        }
        if (opts.is_hot_kernel && k->stats.nr_gprs_32 > AGX_HOT_GPR32_BUDGET) {
            AGX_LOG_WARN("Hot kernel '%s' exceeds %u GPR32 budget (%u GPR32 / %u GPR16, %u:%u spills:fills)",
                         name, AGX_HOT_GPR32_BUDGET, k->stats.nr_gprs_32, k->stats.nr_gprs_16,
                         k->stats.spills, k->stats.fills);
        }
    }

    if (!upload_kernel_to_usc(c->dev, *k, code_bytes.data(), code_bytes.size(),
                              const_bytes.data(), const_bytes.size())) {
        return nullptr;
    }

    if (std::getenv("GGML_AGX_DEBUG") != nullptr || std::getenv("GGML_AGX_SHADERDB") != nullptr) {
        AGX_LOG_INFO("compiled %-32s | %3u GPR32 (%3u h) | %u:%u sp:fi | %4u inst | %5u B | tg=%5u B | dma=%zu%s",
                     name, k->stats.nr_gprs_32, k->stats.nr_gprs_16,
                     k->stats.spills, k->stats.fills, k->stats.instructions,
                     k->binary_size, k->threadgroup_bytes, k->uniform_dma.size(),
                     loaded_from_cache ? " [cached]" : "");
    }

    const agx_kernel * ptr = k.get();
    c->kernels.emplace(name, std::move(k));
    c->order.push_back(name);
    return ptr;
}

const agx_kernel * agx_compiler_compile_glsl(agx_compiler *           c,
                                             const char *             name,
                                             const char *             glsl_source,
                                             const agx_compile_opts & opts) {
    if (!c || !name || !glsl_source) {
        return nullptr;
    }
    if (const agx_kernel * existing = agx_compiler_get(c, name)) {
        return existing;
    }

    std::vector<uint32_t> spv;
    std::string err;
    if (!run_glsl_to_spirv(glsl_source, c->kernel_dir, opts, spv, err)) {
        AGX_LOG_ERROR("Kernel '%s': %s", name, err.c_str());
        return nullptr;
    }
    return agx_compiler_compile_spirv(c, name, spv.data(), spv.size(), opts);
}

const agx_kernel * agx_compiler_compile_file(agx_compiler *           c,
                                             const char *             name,
                                             const char *             path,
                                             const agx_compile_opts & opts) {
    if (!c || !name || !path) {
        return nullptr;
    }
    if (const agx_kernel * existing = agx_compiler_get(c, name)) {
        return existing;
    }

    fs::path p(path);
    if (p.is_relative() && !c->kernel_dir.empty()) {
        p = fs::path(c->kernel_dir) / p;
    }
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) {
        AGX_LOG_ERROR("Kernel '%s': cannot open shader file '%s'", name, p.c_str());
        return nullptr;
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string src = ss.str();

    agx_compile_opts local_opts = opts;
    if (p.has_parent_path()) {
        local_opts.include_dirs.push_back(p.parent_path().string());
    }
    return agx_compiler_compile_glsl(c, name, src.c_str(), local_opts);
}

const agx_kernel * agx_compiler_load_raw_agx3b(agx_compiler * c,
                                               const char *   name,
                                               const void *   code_bytes,
                                               size_t         code_size,
                                               const uint32_t local_size[3],
                                               uint32_t       nr_gprs_16,
                                               uint32_t       staging_halfs,
                                               uint32_t       threadgroup_bytes,
                                               uint32_t       scratch_bytes) {
    if (!c || !name || !code_bytes || code_size == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    if (auto it = c->kernels.find(name); it != c->kernels.end()) {
        return it->second.get();
    }

    auto k = std::make_unique<agx_kernel>();
    k->name              = name;
    k->local_size[0]     = local_size ? std::max(1u, local_size[0]) : 32u;
    k->local_size[1]     = local_size ? std::max(1u, local_size[1]) : 1u;
    k->local_size[2]     = local_size ? std::max(1u, local_size[2]) : 1u;
    k->threadgroup_bytes = threadgroup_bytes;
    k->scratch_bytes     = scratch_bytes;
    k->staging_halfs     = staging_halfs;
    k->uses_root         = true;
    k->root_uniform_half = 0u;
    k->push_offset_bytes = c->vk.abi.push_offset_bytes;
    k->root_size_bytes   = static_cast<uint32_t>(agx_align_up(k->push_offset_bytes + AGX_MAX_PUSH_BYTES, 16u));

    k->stats.nr_gprs_16        = nr_gprs_16;
    k->stats.nr_gprs_32        = (nr_gprs_16 + 1u) / 2u;
    k->stats.staging_halfs     = staging_halfs;
    k->stats.scratch_bytes     = scratch_bytes;
    k->stats.code_bytes        = static_cast<uint32_t>(code_size);
    k->stats.threadgroup_bytes = threadgroup_bytes;
    k->stats.stats_valid       = true;

    if (!upload_kernel_to_usc(c->dev, *k, code_bytes, code_size, nullptr, 0)) {
        return nullptr;
    }

    const agx_kernel * ptr = k.get();
    c->kernels.emplace(name, std::move(k));
    c->order.push_back(name);
    return ptr;
}

bool agx_compiler_audit_all(const agx_compiler * c,
                            bool                 fail_on_spill,
                            uint32_t             max_hot_gpr32) {
    if (!c) return false;
    auto * mc = const_cast<agx_compiler *>(c);
    std::lock_guard<std::mutex> lock(mc->mtx);

    bool ok = true;
    for (const auto & name : mc->order) {
        const auto & k = *mc->kernels.at(name);
        if (!k.stats.stats_valid) continue;
        if (fail_on_spill && (k.stats.spills > 0 || k.stats.fills > 0)) {
            AGX_LOG_ERROR("ShaderDB audit FAIL: '%s' has %u:%u spills:fills",
                          name.c_str(), k.stats.spills, k.stats.fills);
            ok = false;
        }
        if (k.is_hot_kernel && k.stats.nr_gprs_32 > max_hot_gpr32) {
            AGX_LOG_ERROR("ShaderDB audit FAIL: hot kernel '%s' uses %u GPR32 (> %u budget)",
                          name.c_str(), k.stats.nr_gprs_32, max_hot_gpr32);
            ok = false;
        }
    }
    return ok;
}

void agx_compiler_print_shaderdb(const agx_compiler * c) {
    if (!c) return;
    auto * mc = const_cast<agx_compiler *>(c);
    std::lock_guard<std::mutex> lock(mc->mtx);

    std::fprintf(stderr,
                 "\n[ggml-agx] === Borealis AGX3B ShaderDB Audit (%zu kernels) ===\n"
                 "%-34s | %5s | %5s | %7s | %6s | %6s | %6s | %4s\n"
                 "-----------------------------------+-------+-------+---------+--------+--------+--------+-----\n",
                 mc->order.size(), "Kernel", "GPR32", "GPR16", "Sp:Fi", "Instrs", "CodeB", "SRAM", "Hot");
    for (const auto & name : mc->order) {
        const auto & k = *mc->kernels.at(name);
        const bool pass = (k.stats.spills == 0 && k.stats.fills == 0 &&
                           (!k.is_hot_kernel || k.stats.nr_gprs_32 <= AGX_HOT_GPR32_BUDGET));
        std::fprintf(stderr, "%-34s | %5u | %5u | %3u:%-3u | %6u | %6u | %6u | %s %s\n",
                     name.c_str(),
                     k.stats.nr_gprs_32, k.stats.nr_gprs_16,
                     k.stats.spills, k.stats.fills,
                     k.stats.instructions, k.binary_size, k.threadgroup_bytes,
                     k.is_hot_kernel ? "Y" : "N",
                     pass ? "PASS" : "FAIL");
    }
    std::fprintf(stderr, "\n");
}
