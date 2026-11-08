@echo off
setlocal
cd /d "%~dp0.."

set GCC=
set NVCC=
set CUDA_BIN=
set CUDA_LIB=

echo === looking for gcc ===
if exist "%~dp0..\..\tools\llvm-mingw\bin\gcc.exe" set "GCC=%~dp0..\..\tools\llvm-mingw\bin\gcc.exe"
if not defined GCC if exist "%~dp0..\..\tools\llvm-mingw\bin\x86_64-w64-mingw32-gcc.exe" set "GCC=%~dp0..\..\tools\llvm-mingw\bin\x86_64-w64-mingw32-gcc.exe"
if not defined GCC if exist "%~dp0..\..\tools\llvm-mingw-20261104-msvcrt-x86_64\bin\gcc.exe" set "GCC=%~dp0..\..\tools\llvm-mingw-20261104-msvcrt-x86_64\bin\gcc.exe"
if not defined GCC if exist "%~dp0..\..\tools\llvm-mingw-20261104-msvcrt-x86_64\bin\x86_64-w64-mingw32-gcc.exe" set "GCC=%~dp0..\..\tools\llvm-mingw-20261104-msvcrt-x86_64\bin\x86_64-w64-mingw32-gcc.exe"
if not defined GCC where gcc >nul 2>nul && for /f "delims=" %%i in ('where gcc') do set "GCC=%%i"

if not defined GCC goto no_gcc
echo GCC=%GCC%
goto find_nvcc

:no_gcc
echo ERROR: gcc not found.
echo Unzip tools\llvm-mingw-msvcrt-x86_64.zip first.
pause
exit /b 1

:find_nvcc
echo === looking for nvcc ===
where nvcc >nul 2>nul && for /f "delims=" %%i in ('where nvcc') do set "NVCC=%%i"
if not defined NVCC if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc.exe" set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\nvcc.exe"
if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\nvcc.exe" set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\nvcc.exe"
if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe" set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe"

if not defined NVCC goto no_nvcc
echo NVCC=%NVCC%
"%NVCC%" --version
if errorlevel 1 goto no_nvcc
goto have_nvcc

:no_nvcc
echo.
echo ERROR: nvcc not found.
echo.
echo Win7 needs CUDA Toolkit 10.2 installed FIRST:
echo   https://developer.nvidia.com/cuda-10.2-download-archive
echo Choose: Windows / x86_64 / 7 / exe local
echo.
echo After install, either:
echo   1) Open "x64 Native Tools" / CUDA command prompt, OR
echo   2) Add this to PATH then open a NEW cmd:
echo      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin
echo.
echo Also install Visual Studio 2017 Build Tools or VS2015/2017
echo because Windows nvcc requires MSVC cl.exe as host compiler.
echo Then run: scripts\build_win_cuda_msvc.bat
echo.
pause
exit /b 1

:have_nvcc
for %%I in ("%NVCC%") do set "CUDA_BIN=%%~dpI"
set "CUDA_LIB=%CUDA_BIN%..\lib\x64"
if not exist "%CUDA_LIB%\cudart.lib" set "CUDA_LIB=%CUDA_BIN%..\lib"

echo.
echo NOTE: On Windows, nvcc normally needs MSVC.
echo If the next steps fail with cl.exe / compiler errors,
echo install VS2017 Build Tools and use scripts\build_win_cuda_msvc.bat
echo.

echo === compile cuda_kernels.cu ===
"%NVCC%" -O3 -c -Xcompiler "/MD" -o src\cuda_kernels.obj src\cuda_kernels.cu
if errorlevel 1 goto nvcc_fail

echo === compile cuda_fused.cu ===
"%NVCC%" -O3 -c -Xcompiler "/MD" -o src\cuda_fused.obj src\cuda_fused.cu
if errorlevel 1 goto nvcc_fail

echo === mingw link may fail with MSVC .obj; prefer build_win_cuda_msvc.bat ===
"%GCC%" -O3 -Wall -Wno-unused-parameter -fPIC -Ithird_party/vulkan -Isrc -Iinclude -Wno-missing-field-initializers -DHYMT2_CUDA=1 -DHYMT2_NO_VULKAN -D_WIN32_WINNT=0x0601 -DWINVER=0x0601 -o hymt2_cli.exe tools\hymt2_cli.c src\model.c src\kernels.c src\gguf.c src\tokenizer.c src\threadpool.c src\vulkan.c src\device.c src\cuda.c src\npu.c src\metal_stub.c src\cuda_kernels.obj src\cuda_fused.obj -lm -lcudart -static-libgcc -Wl,--major-os-version=6 -Wl,--minor-os-version=1 -Wl,--major-subsystem-version=6 -Wl,--minor-subsystem-version=1 -L"%CUDA_LIB%"
if errorlevel 1 goto link_fail

echo.
echo OK: %CD%\hymt2_cli.exe
echo Copy cudart64_102.dll from %CUDA_BIN% to this folder.
pause
exit /b 0

:nvcc_fail
echo.
echo nvcc compile failed. Install Visual Studio 2017 Build Tools, then run:
echo   scripts\build_win_cuda_msvc.bat
pause
exit /b 1

:link_fail
echo.
echo Link with MinGW failed. Use MSVC script instead:
echo   scripts\build_win_cuda_msvc.bat
pause
exit /b 1
