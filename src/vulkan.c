/* Vulkan 计算后端 (可选): 把大矩阵乘 (STQ1_0/Q2_0C 面板、Q4_K、Q6_K) 放到 GPU 上做。
 * - 运行时 dlopen / LoadLibrary 加载 Vulkan 加载器, 不链接 libvulkan; 没有 Vulkan 时 vk_create 返回 NULL。
 * - 只用 Vulkan 1.0 核心功能 + uint 存储缓冲 (不需要 8/16 位存储扩展), 以便在各种驱动上都能跑。
 * - 单个存储缓冲有范围上限 (规范只保证 128MB), 所以权重按行分块, 每块一个缓冲和一个描述符集。 */
#ifndef HYMT2_NO_VULKAN
#define VK_NO_PROTOTYPES 1
#include "internal.h"
#include <vulkan/vulkan_core.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_shaders.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define VK_FUNCS(X)                                                                                                   \
    X(vkCreateInstance) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)           \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice)              \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements)      \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateShaderModule)                  \
    X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool)             \
    X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool)           \
    X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer)                 \
    X(vkResetCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch)   \
    X(vkCmdCopyBuffer) X(vkQueueSubmit) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)        \
    X(vkDeviceWaitIdle)

enum { KIND_PANEL = 0, KIND_Q4K = 1, KIND_Q6K = 2 };
#define Q6K_PADDED_BYTES 212

typedef struct { VkBuffer wb, sb; VkDeviceMemory wm, sm; VkDescriptorSet ds; uint32_t rows, row0; } vkpart;

struct vkweight {
    vkctx *c;
    int kind, n, k, nb;
    uint32_t shift, off;
    vkpart *parts;
    int np;
    VkBuffer ob; VkDeviceMemory om; float *op;
};

struct vkctx {
    void *lib;
    PFN_vkGetInstanceProcAddr gipa;
#define X(n) PFN_##n n;
    VK_FUNCS(X)
#undef X
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qfam;
    VkPhysicalDeviceMemoryProperties mp; VkPhysicalDeviceProperties props;
    uint64_t max_range;
    VkShaderModule sm[3]; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe[3];
    VkDescriptorPool pool; VkCommandPool cp; VkCommandBuffer cb; VkFence fence;
    VkBuffer act; VkDeviceMemory actm; uint8_t *actp; size_t act_cap;
    VkBuffer dummy; VkDeviceMemory dummym;
    bool software, broken;
};

typedef struct { uint32_t rows, nb, nt, n_total, row0, shift, off, pad0; } pushc;

/* ------------------------------------------------------------------ 小工具 */

static bool load_lib(vkctx *c) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("vulkan-1.dll");
    if (!h) return false;
    c->lib = (void *)h;
    c->gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(h, "vkGetInstanceProcAddr");
#else
    /* Android 优先绝对路径: 裸 adb shell 下短名有时会解析到不可用的 stub。 */
    const char *names[] = {
#if defined(__ANDROID__)
        "/system/lib64/libvulkan.so",
        "/system/lib/libvulkan.so",
#endif
        "libvulkan.so.1", "libvulkan.so", "libvulkan.1.dylib", "libMoltenVK.dylib"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && !c->lib; i++)
        c->lib = dlopen(names[i], RTLD_NOW);
    if (!c->lib) return false;
    c->gipa = (PFN_vkGetInstanceProcAddr)dlsym(c->lib, "vkGetInstanceProcAddr");
#endif
    return c->gipa != NULL;
}

static int find_mem(vkctx *c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c->mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (c->mp.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    return -1;
}

static bool make_buf(vkctx *c, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, VkBuffer *b, VkDeviceMemory *m, void **mapped) {
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size ? size : 16; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (c->vkCreateBuffer(c->dev, &bi, NULL, b) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    c->vkGetBufferMemoryRequirements(c->dev, *b, &mr);
    int mt = find_mem(c, mr.memoryTypeBits, want);
    if (mt < 0) { c->vkDestroyBuffer(c->dev, *b, NULL); *b = VK_NULL_HANDLE; return false; }
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size; ai.memoryTypeIndex = (uint32_t)mt;
    if (c->vkAllocateMemory(c->dev, &ai, NULL, m) != VK_SUCCESS) { c->vkDestroyBuffer(c->dev, *b, NULL); *b = VK_NULL_HANDLE; return false; }
    c->vkBindBufferMemory(c->dev, *b, *m, 0);
    if (mapped && c->vkMapMemory(c->dev, *m, 0, VK_WHOLE_SIZE, 0, mapped) != VK_SUCCESS) return false;
    return true;
}

static void free_buf(vkctx *c, VkBuffer b, VkDeviceMemory m) {
    if (b) c->vkDestroyBuffer(c->dev, b, NULL);
    if (m) c->vkFreeMemory(c->dev, m, NULL);
}

static bool submit_wait(vkctx *c) {
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &c->cb;
    c->vkResetFences(c->dev, 1, &c->fence);
    if (c->vkQueueSubmit(c->q, 1, &si, c->fence) != VK_SUCCESS) return false;
    return c->vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, 30ull * 1000000000ull) == VK_SUCCESS;
}

static bool begin_cb(vkctx *c) {
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    c->vkResetCommandBuffer(c->cb, 0);
    return c->vkBeginCommandBuffer(c->cb, &bi) == VK_SUCCESS;
}

/* ------------------------------------------------------------------ 创建 / 销毁 */

static const char *dev_type_name(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "software";
    default: return "other";
    }
}

void vk_destroy(vkctx *c) {
    if (!c) return;
    if (c->dev) {
        c->vkDeviceWaitIdle(c->dev);
        free_buf(c, c->act, c->actm); free_buf(c, c->dummy, c->dummym);
        if (c->fence) c->vkDestroyFence(c->dev, c->fence, NULL);
        if (c->cp) c->vkDestroyCommandPool(c->dev, c->cp, NULL);
        if (c->pool) c->vkDestroyDescriptorPool(c->dev, c->pool, NULL);
        for (int i = 0; i < 3; i++) {
            if (c->pipe[i]) c->vkDestroyPipeline(c->dev, c->pipe[i], NULL);
            if (c->sm[i]) c->vkDestroyShaderModule(c->dev, c->sm[i], NULL);
        }
        if (c->pl) c->vkDestroyPipelineLayout(c->dev, c->pl, NULL);
        if (c->dsl) c->vkDestroyDescriptorSetLayout(c->dev, c->dsl, NULL);
        c->vkDestroyDevice(c->dev, NULL);
    }
    if (c->inst) c->vkDestroyInstance(c->inst, NULL);
#ifdef _WIN32
    if (c->lib) FreeLibrary((HMODULE)c->lib);
#else
    if (c->lib) dlclose(c->lib);
#endif
    free(c);
}

bool vk_is_software(const vkctx *c) { return c->software; }

static int rank_type(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 1;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 9;
    default: return 5;
    }
}

vkctx *vk_create(const char *pref, char *info, size_t cap) {
    vkctx *c = (vkctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
#define FAILV(...) do { if (info) snprintf(info, cap, __VA_ARGS__); vk_destroy(c); return NULL; } while (0)
    if (!load_lib(c)) FAILV("没有找到 Vulkan 加载器 (libvulkan / vulkan-1.dll)");
    c->vkCreateInstance = (PFN_vkCreateInstance)c->gipa(NULL, "vkCreateInstance");
    if (!c->vkCreateInstance) FAILV("vkCreateInstance 不可用");
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "hymt2"; app.apiVersion = VK_API_VERSION_1_0;
    /* MoltenVK (macOS) 是 portability ICD：必须开枚举扩展，否则 vkCreateInstance 会失败。 */
    const char *inst_exts[2];
    uint32_t n_inst_ext = 0;
#ifdef __APPLE__
    inst_exts[n_inst_ext++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
#endif
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
#ifdef __APPLE__
    ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    if (n_inst_ext) { ici.enabledExtensionCount = n_inst_ext; ici.ppEnabledExtensionNames = inst_exts; }
    if (c->vkCreateInstance(&ici, NULL, &c->inst) != VK_SUCCESS) FAILV("创建 Vulkan 实例失败");
    /* 实例级函数走 gipa(inst); 设备级在 CreateDevice 后再用 gipa 取 (Android Mali 上早期取设备函数可能拿到空壳)。 */
#define X(n) c->n = (PFN_##n)c->gipa(c->inst, #n);
    VK_FUNCS(X)
#undef X
    if (!c->vkEnumeratePhysicalDevices || !c->vkGetPhysicalDeviceProperties || !c->vkDestroyInstance)
        FAILV("缺少 Vulkan 实例函数");
    uint32_t nd = 0;
    c->vkEnumeratePhysicalDevices(c->inst, &nd, NULL);
    if (nd == 0) FAILV("没有 Vulkan 设备");
    VkPhysicalDevice *pds = (VkPhysicalDevice *)calloc(nd, sizeof(*pds));
    c->vkEnumeratePhysicalDevices(c->inst, &nd, pds);

    const bool list = pref && !strcmp(pref, "list"), force = pref && !strcmp(pref, "force");
    const int want_idx = (pref && pref[0] >= '0' && pref[0] <= '9') ? atoi(pref) : -1;
    int best = -1, best_rank = 1000;
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p;
        c->vkGetPhysicalDeviceProperties(pds[i], &p);
        if (list) fprintf(stderr, "[hymt2] Vulkan 设备 %u: %s (%s)\n", i, p.deviceName, dev_type_name(p.deviceType));
        int rk = rank_type(p.deviceType);
        if (want_idx >= 0) { if ((int)i == want_idx) { best = (int)i; best_rank = rk; } continue; }
        if (rk == 9 && !force) continue;           /* 默认不用软件渲染器 */
        if (rk < best_rank) { best_rank = rk; best = (int)i; }
    }
    if (list) { free(pds); FAILV("设备列表已打印"); }
    if (best < 0) { free(pds); FAILV("没有可用的 GPU (只有软件渲染器? 设 HYMT2_VK=force 可强制使用)"); }
    c->pd = pds[best];
    free(pds);
    if (!c->vkGetPhysicalDeviceMemoryProperties || !c->vkGetPhysicalDeviceQueueFamilyProperties)
        FAILV("缺少 Vulkan 物理设备函数");
    c->vkGetPhysicalDeviceProperties(c->pd, &c->props);
    c->vkGetPhysicalDeviceMemoryProperties(c->pd, &c->mp);
    c->software = c->props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    c->max_range = c->props.limits.maxStorageBufferRange;
    if (c->max_range > (1ull << 30)) c->max_range = 1ull << 30;
    if (getenv("HYMT2_VK_MAXRANGE")) c->max_range = (uint64_t)strtoull(getenv("HYMT2_VK_MAXRANGE"), NULL, 10); /* 测试: 强制分块 */
    if (c->props.limits.maxComputeSharedMemorySize < 64 * sizeof(float)) FAILV("共享内存太小");

    uint32_t nq = 0;
    c->vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, NULL);
    VkQueueFamilyProperties *qp = (VkQueueFamilyProperties *)calloc(nq, sizeof(*qp));
    c->vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, qp);
    c->qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c->qfam = i; break; }
    free(qp);
    if (c->qfam == UINT32_MAX) FAILV("设备没有计算队列");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = c->qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
#ifdef __APPLE__
    /* MoltenVK 要求启用 portability subset 才能创建逻辑设备。 */
    const char *dev_exts[] = { "VK_KHR_portability_subset" };
    di.enabledExtensionCount = 1;
    di.ppEnabledExtensionNames = dev_exts;
#endif
    if (!c->vkCreateDevice) FAILV("缺少 vkCreateDevice");
    if (c->vkCreateDevice(c->pd, &di, NULL, &c->dev) != VK_SUCCESS) FAILV("创建 Vulkan 设备失败");
    /* 设备创建后再解析一遍, 覆盖可能的空/过期指针 */
#define X(n) do { PFN_##n p = (PFN_##n)c->gipa(c->inst, #n); if (p) c->n = p; } while (0);
    VK_FUNCS(X)
#undef X
    if (!c->vkGetDeviceQueue) FAILV("缺少 vkGetDeviceQueue");
    c->vkGetDeviceQueue(c->dev, c->qfam, 0, &c->q);
    if (getenv("HYMT2_VERBOSE"))
        fprintf(stderr, "[hymt2] Vulkan 设备就绪: %s (%s)\n", c->props.deviceName, dev_type_name(c->props.deviceType));

    if (!c->vkCreateDescriptorSetLayout || !c->vkCreatePipelineLayout || !c->vkCreateShaderModule ||
        !c->vkCreateComputePipelines || !c->vkCreateDescriptorPool || !c->vkCreateCommandPool ||
        !c->vkAllocateCommandBuffers || !c->vkCreateFence || !c->vkCreateBuffer || !c->vkAllocateMemory)
        FAILV("缺少 Vulkan 设备函数");
    /* 描述符布局: 4 个存储缓冲 (权重 / 缩放 / 激活 / 输出), 推送常量 32 字节 */
    VkDescriptorSetLayoutBinding bind[4];
    for (int i = 0; i < 4; i++) {
        memset(&bind[i], 0, sizeof(bind[i]));
        bind[i].binding = (uint32_t)i; bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].descriptorCount = 1; bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = 4; dli.pBindings = bind;
    if (c->vkCreateDescriptorSetLayout(c->dev, &dli, NULL, &c->dsl) != VK_SUCCESS) FAILV("descriptor layout");
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushc) };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &c->dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    if (c->vkCreatePipelineLayout(c->dev, &pli, NULL, &c->pl) != VK_SUCCESS) FAILV("pipeline layout");
    const uint32_t *code[3] = { spv_panel, spv_q4k, spv_q6k };
    const size_t sizes[3] = { sizeof(spv_panel), sizeof(spv_q4k), sizeof(spv_q6k) };
    for (int i = 0; i < 3; i++) {
        VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smi.codeSize = sizes[i]; smi.pCode = code[i];
        if (c->vkCreateShaderModule(c->dev, &smi, NULL, &c->sm[i]) != VK_SUCCESS) FAILV("创建 shader %d 失败", i);
        VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = c->sm[i]; cpi.stage.pName = "main";
        cpi.layout = c->pl;
        if (c->vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &c->pipe[i]) != VK_SUCCESS) FAILV("创建 pipeline %d 失败", i);
        if (getenv("HYMT2_VERBOSE")) fprintf(stderr, "[hymt2] Vulkan pipeline %d ok\n", i);
    }
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * 4096 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 4096; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    if (c->vkCreateDescriptorPool(c->dev, &dpi, NULL, &c->pool) != VK_SUCCESS) FAILV("descriptor pool");
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpci.queueFamilyIndex = c->qfam;
    if (c->vkCreateCommandPool(c->dev, &cpci, NULL, &c->cp) != VK_SUCCESS) FAILV("command pool");
    VkCommandBufferAllocateInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbi.commandPool = c->cp; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    if (c->vkAllocateCommandBuffers(c->dev, &cbi, &c->cb) != VK_SUCCESS) FAILV("command buffer");
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (c->vkCreateFence(c->dev, &fi, NULL, &c->fence) != VK_SUCCESS) FAILV("fence");

    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    c->act_cap = (size_t)MAX_BATCH * 96 * sizeof(q8k_block); /* 最多 K = 24576 */
    if (!make_buf(c, c->act_cap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, &c->act, &c->actm, (void **)&c->actp)) FAILV("激活缓冲分配失败");
    if (!make_buf(c, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, &c->dummy, &c->dummym, NULL)) FAILV("dummy 缓冲分配失败");
    if (info) snprintf(info, cap, "%s (%s), 单缓冲上限 %llu MB", c->props.deviceName, dev_type_name(c->props.deviceType), (unsigned long long)(c->max_range >> 20));
    return c;
#undef FAILV
}

/* ------------------------------------------------------------------ 权重上传 */

float *vk_weight_out(vkweight *w) { return w ? w->op : NULL; }

void vk_release(vkweight *w) {
    if (!w) return;
    vkctx *c = w->c;
    for (int i = 0; i < w->np; i++) { free_buf(c, w->parts[i].wb, w->parts[i].wm); free_buf(c, w->parts[i].sb, w->parts[i].sm); }
    free_buf(c, w->ob, w->om);
    free(w->parts);
    free(w);
}

static void write_ds(vkctx *c, VkDescriptorSet ds, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3) {
    VkBuffer bufs[4] = { b0, b1, b2, b3 };
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet wr[4];
    for (int i = 0; i < 4; i++) {
        bi[i].buffer = bufs[i]; bi[i].offset = 0; bi[i].range = VK_WHOLE_SIZE;
        memset(&wr[i], 0, sizeof(wr[i]));
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = ds; wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bi[i];
    }
    c->vkUpdateDescriptorSets(c->dev, 4, wr, 0, NULL);
}

vkweight *vk_upload(vkctx *c, const weight_t *w) {
    int kind;
    size_t wrow;   /* 设备端每行字节数 */
    const int nb = w->k / QK_K;
    if ((w->type == GT_STQ1_0 || w->type == GT_Q2_0C) && w->pk) { kind = KIND_PANEL; wrow = (size_t)nb * 64; }
    else if (w->type == GT_Q4_K) { kind = KIND_Q4K; wrow = (size_t)nb * 144; }
    else if (w->type == GT_Q6_K) { kind = KIND_Q6K; wrow = (size_t)nb * Q6K_PADDED_BYTES; }
    else return NULL;
    const size_t srow = kind == KIND_PANEL ? (size_t)nb * 4 : 0;
    const size_t maxrow = wrow > srow ? wrow : srow;
    if (maxrow > c->max_range || c->broken) return NULL;
    size_t rows_per = (size_t)(c->max_range / maxrow);
    if (rows_per > 65535u * 8u) rows_per = 65535u * 8u; /* 一个调度最多 65535 个工作组 */

    vkweight *vw = (vkweight *)calloc(1, sizeof(*vw));
    vw->c = c; vw->kind = kind; vw->n = w->n; vw->k = w->k; vw->nb = nb;
    vw->shift = (uint32_t)w->pshift; vw->off = (uint32_t)w->poff;
    vw->np = (int)(((size_t)w->n + rows_per - 1) / rows_per);
    vw->parts = (vkpart *)calloc((size_t)vw->np, sizeof(vkpart));
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags devl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    void *omap = NULL;
    if (!make_buf(c, (VkDeviceSize)w->n * MAX_BATCH * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, &vw->ob, &vw->om, &omap)) { vk_release(vw); return NULL; }
    vw->op = (float *)omap;

    VkBuffer stg = VK_NULL_HANDLE; VkDeviceMemory stgm = VK_NULL_HANDLE; void *stgp = NULL;
    size_t stg_cap = rows_per * maxrow < (size_t)w->n * maxrow ? rows_per * maxrow : (size_t)w->n * maxrow;
    if (!make_buf(c, stg_cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, &stg, &stgm, &stgp)) { vk_release(vw); return NULL; }
    bool ok = true;
    for (int p = 0; p < vw->np && ok; p++) {
        vkpart *pt = &vw->parts[p];
        pt->row0 = (uint32_t)((size_t)p * rows_per);
        pt->rows = (uint32_t)(((size_t)w->n - pt->row0) < rows_per ? ((size_t)w->n - pt->row0) : rows_per);
        const size_t wbytes = (size_t)pt->rows * wrow, sbytes = (size_t)pt->rows * srow;
        if (!make_buf(c, wbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, devl, &pt->wb, &pt->wm, NULL)) { ok = false; break; }
        if (kind == KIND_PANEL) {
            if (!make_buf(c, sbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, devl, &pt->sb, &pt->sm, NULL)) { ok = false; break; }
        }
        /* 1) 权重: 填 staging, 拷到设备 */
        if (kind == KIND_PANEL) memcpy(stgp, w->pk + (size_t)pt->row0 * wrow, wbytes);
        else if (kind == KIND_Q4K) memcpy(stgp, w->data + (size_t)pt->row0 * w->row_bytes, wbytes);
        else { /* Q6_K: 210 -> 212 字节, 保证每块 4 字节对齐 */
            for (size_t i = 0; i < (size_t)pt->rows * (size_t)nb; i++) {
                const uint8_t *src = w->data + (size_t)pt->row0 * w->row_bytes + i * sizeof(block_q6_k);
                uint8_t *dst = (uint8_t *)stgp + i * Q6K_PADDED_BYTES;
                memcpy(dst, src, sizeof(block_q6_k)); dst[210] = dst[211] = 0;
            }
        }
        if (!begin_cb(c)) { ok = false; break; }
        VkBufferCopy cp = { 0, 0, wbytes };
        c->vkCmdCopyBuffer(c->cb, stg, pt->wb, 1, &cp);
        c->vkEndCommandBuffer(c->cb);
        if (!submit_wait(c)) { ok = false; break; }
        /* 2) 面板缩放 */
        if (kind == KIND_PANEL) {
            memcpy(stgp, w->ps + (size_t)pt->row0 * nb, sbytes);
            if (!begin_cb(c)) { ok = false; break; }
            VkBufferCopy cs = { 0, 0, sbytes };
            c->vkCmdCopyBuffer(c->cb, stg, pt->sb, 1, &cs);
            c->vkEndCommandBuffer(c->cb);
            if (!submit_wait(c)) { ok = false; break; }
        }
        VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool = c->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &c->dsl;
        if (c->vkAllocateDescriptorSets(c->dev, &dai, &pt->ds) != VK_SUCCESS) { ok = false; break; }
        write_ds(c, pt->ds, pt->wb, kind == KIND_PANEL ? pt->sb : c->dummy, c->act, vw->ob);
    }
    free_buf(c, stg, stgm);
    if (!ok) { vk_release(vw); return NULL; }
    return vw;
}

/* ------------------------------------------------------------------ 运行 */

bool vk_matmul(vkctx *c, vkweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) {
    if (c->broken || nt < 1 || nt > MAX_BATCH) return false;
    const int nb = ws[0]->nb;
    for (int i = 1; i < cnt; i++) if (ws[i]->nb != nb) return false;
    const size_t abytes = (size_t)nt * (size_t)nb * sizeof(q8k_block);
    if (abytes > c->act_cap) return false;
    memcpy(c->actp, xq, abytes);
    if (!begin_cb(c)) { c->broken = true; return false; }
    int last = -1;
    for (int i = 0; i < cnt; i++) {
        vkweight *w = ws[i];
        if (w->kind != last) { c->vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe[w->kind]); last = w->kind; }
        for (int p = 0; p < w->np; p++) {
            vkpart *pt = &w->parts[p];
            pushc pc = { pt->rows, (uint32_t)nb, (uint32_t)nt, (uint32_t)w->n, pt->row0, w->shift, w->off, 0 };
            c->vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &pt->ds, 0, NULL);
            c->vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            c->vkCmdDispatch(c->cb, (pt->rows + 7u) / 8u, (uint32_t)nt, 1);
        }
    }
    c->vkEndCommandBuffer(c->cb);
    if (!submit_wait(c)) { c->broken = true; return false; }
    /* host-coherent out: wait 后 CPU 可见; 别名 outs 时零拷贝 */
    for (int i = 0; i < cnt; i++) {
        if (!outs[i] || outs[i] == ws[i]->op) continue;
        memcpy(outs[i], ws[i]->op, (size_t)ws[i]->n * (size_t)nt * sizeof(float));
    }
    return true;
}

#else /* HYMT2_NO_VULKAN: 空实现, 让其余代码无需 #ifdef */
#include "internal.h"
#include <stdio.h>
vkctx *vk_create(const char *pref, char *info, size_t cap) { (void)pref; if (info) snprintf(info, cap, "本构建不含 Vulkan"); return NULL; }
bool vk_is_software(const vkctx *c) { (void)c; return false; }
void vk_destroy(vkctx *c) { (void)c; }
vkweight *vk_upload(vkctx *c, const weight_t *w) { (void)c; (void)w; return NULL; }
void vk_release(vkweight *w) { (void)w; }
float *vk_weight_out(vkweight *w) { (void)w; return NULL; }
bool vk_matmul(vkctx *c, vkweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) { (void)c; (void)ws; (void)outs; (void)cnt; (void)xq; (void)nt; return false; }
#endif
