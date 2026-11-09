@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

rem Prefer clang-cl: source uses GCC/Clang target multiversion + builtins.
rem nvcc still needs MSVC for .cu host; clang-cl links with the same link.exe.
where clang-cl >nul 2>nul
if errorlevel 1 (
  echo ERROR: clang-cl not on PATH. Install LLVM or use a runner with LLVM.
  exit /b 1
)
where cl >nul 2>nul
if errorlevel 1 (
  echo ERROR: cl.exe not on PATH ^(needed by nvcc host^)
  exit /b 1
)
where nvcc >nul 2>nul
if errorlevel 1 (
  echo ERROR: nvcc not on PATH
  exit /b 1
)

for /f "delims=" %%i in ('where nvcc') do set "NVCC=%%i"
for %%I in ("%NVCC%") do set "CUDA_BIN=%%~dpI"
set "CUDA_LIB=%CUDA_BIN%..\lib\x64"
set "CUDA_INC=%CUDA_BIN%..\include"
if not exist "%CUDA_LIB%\cudart.lib" set "CUDA_LIB=%CUDA_BIN%..\lib"

echo NVCC=%NVCC%
nvcc --version
clang-cl --version

if not exist build_msvc mkdir build_msvc

rem Universal: CUDA + Vulkan + CPU. Vulkan = LoadLibrary(vulkan-1.dll).
rem CUDA delay-load so missing cudart still allows Vulkan/CPU.
set CFLAGS=/nologo /O2 /clang:-O3 /MD /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /DHYMT2_CUDA=1 /D_WIN32_WINNT=0x0601 /DWINVER=0x0601 /Ithird_party\vulkan /Isrc /Iinclude /I"%CUDA_INC%"

echo === nvcc ===
nvcc -O3 -c -Xcompiler "/MD /nologo" -o build_msvc\cuda_kernels.obj src\cuda_kernels.cu
if errorlevel 1 exit /b 1
nvcc -O3 -c -Xcompiler "/MD /nologo" -o build_msvc\cuda_fused.obj src\cuda_fused.cu
if errorlevel 1 exit /b 1

echo === clang-cl ===
clang-cl %CFLAGS% /c /Fobuild_msvc\hymt2_cli.obj tools\hymt2_cli.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\model.obj src\model.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\kernels.obj src\kernels.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\gguf.obj src\gguf.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\tokenizer.obj src\tokenizer.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\threadpool.obj src\threadpool.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\vulkan.obj src\vulkan.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\device.obj src\device.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\cuda.obj src\cuda.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\npu.obj src\npu.c
if errorlevel 1 exit /b 1
clang-cl %CFLAGS% /c /Fobuild_msvc\metal_stub.obj src\metal_stub.c
if errorlevel 1 exit /b 1

echo === resolve cudart dll name for DELAYLOAD ===
set "CUDART_DLL="
for %%F in ("%CUDA_BIN%cudart64_*.dll") do set "CUDART_DLL=%%~nxF"
if "%CUDART_DLL%"=="" (
  echo ERROR: no cudart64_*.dll next to nvcc
  exit /b 1
)
echo CUDART_DLL=%CUDART_DLL%

echo === link ===
link /nologo /SUBSYSTEM:CONSOLE,6.01 /OUT:hymt2_cli.exe ^
  build_msvc\hymt2_cli.obj build_msvc\model.obj build_msvc\kernels.obj build_msvc\gguf.obj ^
  build_msvc\tokenizer.obj build_msvc\threadpool.obj build_msvc\vulkan.obj build_msvc\device.obj ^
  build_msvc\cuda.obj build_msvc\npu.obj build_msvc\metal_stub.obj ^
  build_msvc\cuda_kernels.obj build_msvc\cuda_fused.obj ^
  cudart.lib delayimp.lib /DELAYLOAD:%CUDART_DLL% /LIBPATH:"%CUDA_LIB%"
if errorlevel 1 exit /b 1

copy /Y "%CUDA_BIN%%CUDART_DLL%" .
dir /b hymt2_cli.exe %CUDART_DLL%
echo BUILD OK (CUDA+Vulkan+CPU universal; HYMT2_DEVICE=auto)
endlocal
