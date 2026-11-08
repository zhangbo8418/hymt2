@echo off
setlocal
cd /d "%~dp0.."

echo === hymt2 CUDA fused build via MSVC + nvcc ===
echo Run this from "x64 Native Tools Command Prompt for VS 2017"
echo or a cmd where both cl.exe and nvcc.exe are on PATH.
echo.

where cl >nul 2>nul
if errorlevel 1 (
  echo ERROR: cl.exe not found.
  echo Install Visual Studio 2017 Build Tools with "VC++ 2017 toolset",
  echo then open "x64 Native Tools Command Prompt for VS 2017".
  pause
  exit /b 1
)

where nvcc >nul 2>nul
if errorlevel 1 (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc.exe" set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin;%PATH%"
)
where nvcc >nul 2>nul
if errorlevel 1 (
  echo ERROR: nvcc not found. Install CUDA Toolkit 10.2.
  echo https://developer.nvidia.com/cuda-10.2-download-archive
  pause
  exit /b 1
)

for /f "delims=" %%i in ('where nvcc') do set "NVCC=%%i"
for %%I in ("%NVCC%") do set "CUDA_BIN=%%~dpI"
set "CUDA_LIB=%CUDA_BIN%..\lib\x64"
set "CUDA_INC=%CUDA_BIN%..\include"
echo NVCC=%NVCC%
echo CL:
cl 2>&1 | findstr /i "Version"
nvcc --version

if not exist build_msvc mkdir build_msvc
del /q build_msvc\*.obj build_msvc\*.exe 2>nul

rem Universal: CUDA + Vulkan (runtime LoadLibrary) + CPU. Do not define HYMT2_NO_VULKAN.
set CFLAGS=/nologo /O2 /MD /W3 /D_CRT_SECURE_NO_WARNINGS /DHYMT2_CUDA=1 /D_WIN32_WINNT=0x0601 /DWINVER=0x0601 /Ithird_party\vulkan /Isrc /Iinclude /I"%CUDA_INC%"

echo === nvcc kernels ===
nvcc -O3 -c -Xcompiler "/MD /nologo" -o build_msvc\cuda_kernels.obj src\cuda_kernels.cu
if errorlevel 1 goto fail
nvcc -O3 -c -Xcompiler "/MD /nologo" -o build_msvc\cuda_fused.obj src\cuda_fused.cu
if errorlevel 1 goto fail

echo === cl sources ===
cl %CFLAGS% /c /Fobuild_msvc\hymt2_cli.obj tools\hymt2_cli.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\model.obj src\model.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\kernels.obj src\kernels.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\gguf.obj src\gguf.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\tokenizer.obj src\tokenizer.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\threadpool.obj src\threadpool.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\vulkan.obj src\vulkan.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\device.obj src\device.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\cuda.obj src\cuda.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\npu.obj src\npu.c
if errorlevel 1 goto fail
cl %CFLAGS% /c /Fobuild_msvc\metal_stub.obj src\metal_stub.c
if errorlevel 1 goto fail

set "CUDART_DLL="
for %%F in ("%CUDA_BIN%cudart64_*.dll") do set "CUDART_DLL=%%~nxF"
if "%CUDART_DLL%"=="" set "CUDART_DLL=cudart64_102.dll"

echo === link ===
link /nologo /SUBSYSTEM:CONSOLE,6.01 /OUT:hymt2_cli.exe ^
  build_msvc\hymt2_cli.obj build_msvc\model.obj build_msvc\kernels.obj ^
  build_msvc\gguf.obj build_msvc\tokenizer.obj build_msvc\threadpool.obj ^
  build_msvc\vulkan.obj build_msvc\device.obj build_msvc\cuda.obj ^
  build_msvc\npu.obj build_msvc\metal_stub.obj ^
  build_msvc\cuda_kernels.obj build_msvc\cuda_fused.obj ^
  cudart.lib delayimp.lib /DELAYLOAD:%CUDART_DLL% /LIBPATH:"%CUDA_LIB%"
if errorlevel 1 goto fail

copy /Y "%CUDA_BIN%%CUDART_DLL%" . >nul 2>nul

echo.
echo OK: %CD%\hymt2_cli.exe  (CUDA+Vulkan+CPU; default HYMT2_DEVICE=auto)
echo Ship %CUDART_DLL% next to the exe when testing CUDA.
echo GPU driver provides vulkan-1.dll for Vulkan path.
echo.
echo Test:
echo   set HYMT2_DEVICE=auto
echo   set HYMT2_VERBOSE=1
echo   hymt2_cli.exe MODEL.gguf Chinese "Hi" 0 16
pause
exit /b 0

:fail
echo BUILD FAILED
pause
exit /b 1
