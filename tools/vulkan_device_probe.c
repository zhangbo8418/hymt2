/* Android 真机 Vulkan 探测 (计算队列)。交叉编译后 adb push 运行。 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES 1
#include <vulkan/vulkan_core.h>

int main(void) {
    const char *libs[] = {
        "/system/lib64/libvulkan.so",
        "libvulkan.so",
        "/vendor/lib64/hw/vulkan.mali.so",
        NULL
    };
    void *lib = NULL;
    const char *used = NULL;
    for (int i = 0; libs[i]; i++) {
        lib = dlopen(libs[i], RTLD_NOW);
        printf("dlopen %s -> %p (%s)\n", libs[i], lib, lib ? "ok" : dlerror());
        if (lib) { used = libs[i]; break; }
    }
    if (!lib) return 1;

    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    printf("gipa=%p from %s\n", (void *)gipa, used);
    if (!gipa) return 2;

    PFN_vkCreateInstance create = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    printf("vkCreateInstance=%p\n", (void *)create);
    if (!create) return 3;

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "vkprobe";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VkResult rc = create(&ici, NULL, &inst);
    printf("CreateInstance rc=%d inst=%p\n", (int)rc, (void *)inst);
    if (rc != VK_SUCCESS || !inst) return 4;

    PFN_vkEnumeratePhysicalDevices enum_pd =
        (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties get_props =
        (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties get_qf =
        (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkDestroyInstance destroy = (PFN_vkDestroyInstance)gipa(inst, "vkDestroyInstance");
    printf("enum=%p props=%p qf=%p destroy=%p\n", (void *)enum_pd, (void *)get_props, (void *)get_qf, (void *)destroy);
    if (!enum_pd || !get_props) return 5;

    uint32_t nd = 0;
    rc = enum_pd(inst, &nd, NULL);
    printf("EnumeratePhysicalDevices count rc=%d nd=%u\n", (int)rc, nd);
    if (nd == 0) { if (destroy) destroy(inst, NULL); return 6; }

    VkPhysicalDevice *pds = (VkPhysicalDevice *)calloc(nd, sizeof(*pds));
    enum_pd(inst, &nd, pds);
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p;
        get_props(pds[i], &p);
        printf("  dev[%u] name=%s type=%d api=%u.%u.%u\n", i, p.deviceName, (int)p.deviceType,
               VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion));
        if (get_qf) {
            uint32_t nq = 0;
            get_qf(pds[i], &nq, NULL);
            VkQueueFamilyProperties *qf = (VkQueueFamilyProperties *)calloc(nq, sizeof(*qf));
            get_qf(pds[i], &nq, qf);
            for (uint32_t q = 0; q < nq; q++)
                printf("    qfam[%u] flags=0x%x count=%u\n", q, qf[q].queueFlags, qf[q].queueCount);
            free(qf);
        }
    }
    free(pds);
    if (destroy) destroy(inst, NULL);
    dlclose(lib);
    printf("vulkan_device_probe ok\n");
    return 0;
}
