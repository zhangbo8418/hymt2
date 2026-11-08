@echo off
setlocal
cd /d "%~dp0.."

where cl >nul 2>nul
if errorlevel 1 (
  echo ERROR: cl.exe not on PATH
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

if not exist build_msvc mkdir build_msvc

set CFLAGS=/nologo /O2 /MD /W3 /D_CRT_SECURE_NO_WARNINGS /DHYMT2_CUDA=1 /DHYMT2_NO_VULKAN /D_WIN32_WINNT=0x0601 /DWINVER=0x0601 /Ithird_party\vulkan /Isrc /Iinclude /I"%CUDA_INC%"

echo === nvcc ===
nvcc -O3 -c -Xcompiler "/MD /nologo" -o build_msvc\cuda_kernels.obj src\cuda_kernels.cu
if errorlevel 1 exit /b 1
nvcc -O3 -c -Xcompiler "/MD /nologo" -o build_msvc\cuda_fused.obj src\cuda_fused.cu
if errorlevel 1 exit /b 1

echo === cl ===
cl %CFLAGS% /c /Fobuild_msvc\hymt2_cli.obj tools\hymt2_cli.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\model.obj src\model.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\kernels.obj src\kernels.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\gguf.obj src\gguf.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\tokenizer.obj src\tokenizer.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\threadpool.obj src\threadpool.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\vulkan.obj src\vulkan.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\device.obj src\device.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\cuda.obj src\cuda.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\npu.obj src\npu.c
if errorlevel 1 exit /b 1
cl %CFLAGS% /c /Fobuild_msvc\metal_stub.obj src\metal_stub.c
if errorlevel 1 exit /b 1

echo === link ===
link /nologo /SUBSYSTEM:CONSOLE,6.01 /OUT:hymt2_cli.exe build_msvc\hymt2_cli.obj build_msvc\model.obj build_msvc\kernels.obj build_msvc\gguf.obj build_msvc\tokenizer.obj build_msvc\threadpool.obj build_msvc\vulkan.obj build_msvc\device.obj build_msvc\cuda.obj build_msvc\npu.obj build_msvc\metal_stub.obj build_msvc\cuda_kernels.obj build_msvc\cuda_fused.obj cudart.lib /LIBPATH:"%CUDA_LIB%"
if errorlevel 1 exit /b 1

copy /Y "%CUDA_BIN%cudart64_*.dll" .
dir /b hymt2_cli.exe cudart64_*.dll
echo BUILD OK
endlocal
