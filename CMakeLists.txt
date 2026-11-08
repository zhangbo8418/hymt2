# Android NDK / iOS / 其它 CMake 场景 (桌面用 Makefile)。livesub 内调试用。
cmake_minimum_required(VERSION 3.16)
project(hymt2 C OBJC)
set(CMAKE_C_STANDARD 11)

set(HYMT2_SRC
  src/model.c src/kernels.c src/gguf.c src/tokenizer.c src/threadpool.c
  src/vulkan.c src/device.c src/cuda.c src/npu.c src/capi.c)

option(HYMT2_VULKAN "Vulkan backend" ON)
option(HYMT2_METAL "Metal backend (Apple)" ON)
option(HYMT2_CUDA "CUDA backend (needs nvcc)" OFF)
option(HYMT2_DOTPROD "NEON dotprod" OFF)

if(APPLE AND HYMT2_METAL)
  list(APPEND HYMT2_SRC src/metal.m)
  add_compile_definitions(HYMT2_METAL=1)
else()
  list(APPEND HYMT2_SRC src/metal_stub.c)
endif()

if(NOT HYMT2_VULKAN)
  add_compile_definitions(HYMT2_NO_VULKAN)
endif()

add_library(hymt2 ${HYMT2_SRC})
target_include_directories(hymt2 PUBLIC include PRIVATE src third_party/vulkan)
target_compile_options(hymt2 PRIVATE -O3 -fvisibility=hidden)

if(APPLE AND HYMT2_METAL)
  target_link_libraries(hymt2 PRIVATE "-framework Metal" "-framework Foundation")
  target_compile_options(hymt2 PRIVATE -fobjc-arc)
endif()

if(HYMT2_CUDA)
  enable_language(CUDA)
  add_compile_definitions(HYMT2_CUDA=1)
  target_sources(hymt2 PRIVATE src/cuda_kernels.cu src/cuda_fused.cu)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(hymt2 PRIVATE CUDA::cudart)
endif()

find_library(M_LIB m)
if(M_LIB)
  target_link_libraries(hymt2 PRIVATE ${M_LIB})
endif()
find_package(Threads)
if(Threads_FOUND)
  target_link_libraries(hymt2 PRIVATE Threads::Threads)
endif()
if(NOT WIN32 AND NOT APPLE)
  target_link_libraries(hymt2 PRIVATE dl)
endif()

if(HYMT2_DOTPROD)
  target_compile_options(hymt2 PRIVATE -march=armv8.2-a+dotprod)
endif()
