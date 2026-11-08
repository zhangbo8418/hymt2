# hymt2: Hy-MT2 推理库 (CPU + Vulkan + 可选 Metal/CUDA + NPU probe)
#   make                 默认库 + CLI
#   make CUDA=1          需 nvcc + CUDA toolkit
#   make METAL=0         Darwin 上关闭 Metal
#   make VULKAN=0        不含 Vulkan
CC      ?= cc
NVCC    ?= nvcc
CFLAGS  ?= -O3
CFLAGS  += -Wall -Wextra -Wno-unused-parameter -fPIC -fvisibility=hidden
ifdef J4125
CFLAGS  += -mtune=goldmont-plus
endif
ifdef WIN7
CFLAGS  += -D_WIN32_WINNT=0x0601 -DWINVER=0x0601
LDWIN7   = -static-libgcc -Wl,--major-os-version=6 -Wl,--minor-os-version=1 -Wl,--major-subsystem-version=6 -Wl,--minor-subsystem-version=1
endif

CSRC     = src/model.c src/kernels.c src/gguf.c src/tokenizer.c src/threadpool.c src/vulkan.c \
           src/device.c src/cuda.c src/npu.c src/capi.c
OBJC_SRC =
CUDA_OBJ =

CFLAGS  += -Ithird_party/vulkan -Isrc -Wno-missing-field-initializers
ifeq ($(VULKAN),0)
CFLAGS  += -DHYMT2_NO_VULKAN
endif

UNAME   := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME),Darwin)
  EXT = dylib
  SHARED = -dynamiclib
  ifneq ($(METAL),0)
    CFLAGS  += -DHYMT2_METAL=1
    OBJC_SRC = src/metal.m
    LIBS_METAL = -framework Metal -framework Foundation
  else
    CSRC += src/metal_stub.c
  endif
else ifneq (,$(findstring MINGW,$(UNAME))$(findstring MSYS,$(UNAME))$(findstring CYGWIN,$(UNAME)))
  EXT = dll
  SHARED = -shared
  EXE = .exe
  CSRC += src/metal_stub.c
else ifeq ($(OS),Windows_NT)
  EXT = dll
  SHARED = -shared
  EXE = .exe
  CSRC += src/metal_stub.c
else
  EXT = so
  SHARED = -shared
  CSRC += src/metal_stub.c
endif

ifeq ($(CUDA),1)
CFLAGS  += -DHYMT2_CUDA=1
CUDA_OBJ = src/cuda_kernels.o src/cuda_fused.o
LIBS_CUDA = -lcudart
endif

LIBS = -lm -lpthread -ldl $(LIBS_METAL) $(LIBS_CUDA)
ifeq ($(EXT),dll)
  LIBS = -lm $(LIBS_CUDA)
endif
ifeq ($(UNAME),Darwin)
  ifeq ($(CUDA),1)
    LIBS = -lm -lpthread $(LIBS_METAL) $(LIBS_CUDA)
  else
    LIBS = -lm -lpthread $(LIBS_METAL)
  endif
endif

all: libhymt2.$(EXT) hymt2_cli$(EXE)

src/cuda_kernels.o: src/cuda_kernels.cu
	$(NVCC) -O3 -c -Xcompiler -fPIC -o $@ $<

src/cuda_fused.o: src/cuda_fused.cu
	$(NVCC) -O3 -c -Xcompiler -fPIC -o $@ $<

static: hymt2_cli_static
hymt2_cli_static: tools/hymt2_cli.c $(filter-out src/capi.c,$(CSRC)) $(OBJC_SRC) include/hymt2.h
	$(CC) $(CFLAGS) -DHYMT2_NO_VULKAN -static -o $@ tools/hymt2_cli.c $(filter-out src/capi.c,$(CSRC)) $(OBJC_SRC) -lm -lpthread

libhymt2.$(EXT): $(CSRC) $(OBJC_SRC) $(CUDA_OBJ) src/internal.h include/hymt2.h src/unicode_tables.h src/device.h src/metal_shaders.h src/metal_fused_shaders.h
	$(CC) $(CFLAGS) $(SHARED) -o $@ $(CSRC) $(OBJC_SRC) $(CUDA_OBJ) $(LIBS) $(LDWIN7) $(if $(OBJC_SRC),-fobjc-arc)

hymt2_cli$(EXE): tools/hymt2_cli.c $(filter-out src/capi.c,$(CSRC)) $(OBJC_SRC) $(CUDA_OBJ) include/hymt2.h src/metal_shaders.h src/metal_fused_shaders.h
	$(CC) $(CFLAGS) -o $@ tools/hymt2_cli.c $(filter-out src/capi.c,$(CSRC)) $(OBJC_SRC) $(CUDA_OBJ) $(LIBS) $(LDWIN7) $(if $(OBJC_SRC),-fobjc-arc)

test_kernels$(EXE): tests/test_kernels.c src/kernels.c src/internal.h
	$(CC) $(CFLAGS) -Wno-misleading-indentation -o $@ tests/test_kernels.c src/kernels.c -lm

test: test_kernels$(EXE)
	./test_kernels$(EXE)

bench_kernels$(EXE): tests/bench_kernels.c src/kernels.c src/internal.h
	$(CC) $(CFLAGS) -o $@ tests/bench_kernels.c src/kernels.c -lm

bench: bench_kernels$(EXE)
	./bench_kernels$(EXE)

src/unicode_tables.h: tools/gen_unicode.py
	python3 tools/gen_unicode.py $@

clean:
	rm -f libhymt2.so libhymt2.dylib libhymt2.dll hymt2_cli hymt2_cli.exe hymt2_cli_static \
	      test_kernels test_kernels.exe bench_kernels bench_kernels.exe src/cuda_kernels.o src/cuda_fused.o

.PHONY: all static test bench clean
