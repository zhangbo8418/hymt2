@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

where clang-cl >nul 2>nul
if errorlevel 1 (
  echo ERROR: clang-cl not on PATH
  exit /b 1
)

echo clang-cl:
clang-cl --version

if not exist build_msvc mkdir build_msvc

rem Win7-compatible CPU + Vulkan (no CUDA). Vulkan via LoadLibrary(vulkan-1.dll).
rem Match Makefile -O3; /O2 left measurable CPU decode on the table vs original zip.
set CFLAGS=/nologo /O2 /clang:-O3 /MD /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /D_WIN32_WINNT=0x0601 /DWINVER=0x0601 /Ithird_party\vulkan /Isrc /Iinclude

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

echo === link ===
link /nologo /SUBSYSTEM:CONSOLE,6.01 /OUT:hymt2_cli.exe ^
  build_msvc\hymt2_cli.obj build_msvc\model.obj build_msvc\kernels.obj build_msvc\gguf.obj ^
  build_msvc\tokenizer.obj build_msvc\threadpool.obj build_msvc\vulkan.obj build_msvc\device.obj ^
  build_msvc\cuda.obj build_msvc\npu.obj build_msvc\metal_stub.obj
if errorlevel 1 exit /b 1

dir /b hymt2_cli.exe
echo BUILD OK (CPU+Vulkan Win7; HYMT2_DEVICE=auto)
endlocal
