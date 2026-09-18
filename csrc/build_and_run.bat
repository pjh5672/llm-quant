@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
  echo Failed to initialize MSVC environment via vcvarsall.bat
  exit /b 1
)

nvcc -O3 -std=c++17 --generate-code arch=compute_120,code=sm_120 -o w4a8_test.exe w4a8_rtn_naive.cu
if errorlevel 1 (
  echo nvcc compilation failed
  exit /b 1
)

w4a8_test.exe
