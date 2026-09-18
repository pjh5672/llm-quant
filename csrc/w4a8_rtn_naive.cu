// W4A8 symmetric RTN quantization — Milestone 1: naive kernel + self-contained
// correctness harness. Mirrors the pipeline documented in ../w4a8_rtn_notes.md
// (fake quant -> real quant -> int4 nibble packing -> int8 x int4 kernel).
//
// Build (from this directory):
//   build_and_run.bat
// or manually, without the batch script:
//   nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" ^
//        -O3 -std=c++17 --generate-code arch=compute_120,code=sm_120 -o w4a8_test.exe w4a8_rtn_naive.cu
//   w4a8_test.exe
//
// TODO(next milestone): tile over K with shared memory / tensor cores instead
// of the one-thread-per-output-element naive kernel below.
// TODO(next milestone): sweep group_size and per-channel (group_size == K).
// TODO(next milestone): real model weights + PPL instead of synthetic data.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err__));                               \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr int kQMin = -8;
constexpr int kQMax = 7;

template <typename T>
T clampT(T v, T lo, T hi) {
  return std::max(lo, std::min(hi, v));
}

// Shared by CPU simulate and GPU kernel so the two can never drift apart.
// packedRow points at the start of this row's packed bytes (i.e. Wpacked + n*K/2).
inline __host__ __device__ int unpackNibble(const uint8_t* packedRow, int k) {
  uint8_t byte = packedRow[k / 2];
  uint8_t nibble = (k % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
  return static_cast<int>(nibble) - 8;
}

// ---- Stage 1: fake quant RTN ----
// W: [N,K] row-major. Wdq: [N,K] out. scale: [N,nGroups] out.
void fakeQuantRtn(const float* W, int N, int K, int groupSize, float* Wdq,
                   float* scale) {
  int nGroups = K / groupSize;
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < nGroups; ++g) {
      int k0 = g * groupSize, k1 = k0 + groupSize;
      float absMax = 0.0f;
      for (int k = k0; k < k1; ++k)
        absMax = std::max(absMax, std::fabs(W[n * K + k]));
      float s = std::max(absMax / kQMax, 1e-8f);
      scale[n * nGroups + g] = s;
      for (int k = k0; k < k1; ++k) {
        float q = std::round(W[n * K + k] / s);
        q = clampT(q, static_cast<float>(kQMin), static_cast<float>(kQMax));
        Wdq[n * K + k] = q * s;
      }
    }
  }
}

// ---- Stage 2: real quant RTN ----
// Wint: [N,K] int4 values stored in int8_t containers (-8..7).
void realQuantRtn(const float* W, int N, int K, int groupSize, int8_t* Wint,
                   float* scale) {
  int nGroups = K / groupSize;
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < nGroups; ++g) {
      int k0 = g * groupSize, k1 = k0 + groupSize;
      float absMax = 0.0f;
      for (int k = k0; k < k1; ++k)
        absMax = std::max(absMax, std::fabs(W[n * K + k]));
      float s = std::max(absMax / kQMax, 1e-8f);
      scale[n * nGroups + g] = s;
      for (int k = k0; k < k1; ++k) {
        float q = std::round(W[n * K + k] / s);
        q = clampT(q, static_cast<float>(kQMin), static_cast<float>(kQMax));
        Wint[n * K + k] = static_cast<int8_t>(q);
      }
    }
  }
}

// ---- Stage 3: packing ----
// Wint: [N,K] -> packed: [N,K/2]. packed = (high<<4)|low, high/low = q+8.
void packW4(const int8_t* Wint, int N, int K, uint8_t* packed) {
  int halfK = K / 2;
  for (int n = 0; n < N; ++n) {
    for (int j = 0; j < halfK; ++j) {
      uint8_t low = static_cast<uint8_t>(Wint[n * K + 2 * j] + 8);
      uint8_t high = static_cast<uint8_t>(Wint[n * K + 2 * j + 1] + 8);
      packed[n * halfK + j] = static_cast<uint8_t>((high << 4) | low);
    }
  }
}

void unpackW4(const uint8_t* packed, int N, int K, int8_t* Wout) {
  int halfK = K / 2;
  for (int n = 0; n < N; ++n) {
    const uint8_t* row = packed + n * halfK;
    for (int k = 0; k < K; ++k)
      Wout[n * K + k] = static_cast<int8_t>(unpackNibble(row, k));
  }
}

// ---- Runtime activation quant: per-token dynamic int8 (no packing needed) ----
void quantizeActivationInt8(const float* X, int M, int K, int8_t* Xint8,
                             float* xScale) {
  for (int m = 0; m < M; ++m) {
    float absMax = 0.0f;
    for (int k = 0; k < K; ++k)
      absMax = std::max(absMax, std::fabs(X[m * K + k]));
    float s = std::max(absMax / 127.0f, 1e-8f);
    xScale[m] = s;
    for (int k = 0; k < K; ++k) {
      float q = std::round(X[m * K + k] / s);
      q = clampT(q, -128.0f, 127.0f);
      Xint8[m * K + k] = static_cast<int8_t>(q);
    }
  }
}

// fp32 ground truth: Y = X @ W^T. X:[M,K], W:[N,K], Y:[M,N].
void cpuReferenceMatmul(const float* X, const float* W, int M, int N, int K,
                         float* Y) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += X[m * K + k] * W[n * K + k];
      Y[m * N + n] = acc;
    }
  }
}

// CPU re-implementation of the exact GPU kernel algorithm, used to isolate
// "kernel implementation bug" from "expected RTN quantization error".
void cpuW4A8Simulate(const int8_t* Xint8, const float* xScale,
                      const uint8_t* Wpacked, const float* wScale, int M,
                      int N, int K, int groupSize, float* Y) {
  int nGroups = K / groupSize;
  int halfK = K / 2;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      const uint8_t* wRow = Wpacked + n * halfK;
      const int8_t* xRow = Xint8 + m * K;
      float accF = 0.0f;
      for (int g = 0; g < nGroups; ++g) {
        int32_t groupAcc = 0;
        int k0 = g * groupSize, k1 = k0 + groupSize;
        for (int k = k0; k < k1; ++k)
          groupAcc += static_cast<int32_t>(xRow[k]) * unpackNibble(wRow, k);
        accF += static_cast<float>(groupAcc) * wScale[n * nGroups + g];
      }
      Y[m * N + n] = accF * xScale[m];
    }
  }
}

}  // namespace

// ---- Stage 4: naive W4A8 GEMM kernel (one thread per output element) ----
__global__ void w4a8GemmNaiveKernel(const int8_t* __restrict__ Xint8,
                                     const float* __restrict__ xScale,
                                     const uint8_t* __restrict__ Wpacked,
                                     const float* __restrict__ wScale,
                                     float* __restrict__ Y, int M, int N,
                                     int K, int groupSize, int nGroups) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  int halfK = K / 2;
  const uint8_t* wRow = Wpacked + static_cast<size_t>(n) * halfK;
  const int8_t* xRow = Xint8 + static_cast<size_t>(m) * K;

  float accF = 0.0f;
  for (int g = 0; g < nGroups; ++g) {
    int32_t groupAcc = 0;
    int k0 = g * groupSize, k1 = k0 + groupSize;
    for (int k = k0; k < k1; ++k)
      groupAcc += static_cast<int32_t>(xRow[k]) * unpackNibble(wRow, k);
    accF += static_cast<float>(groupAcc) * wScale[n * nGroups + g];
  }
  Y[static_cast<size_t>(m) * N + n] = accF * xScale[m];
}

int main() {
  const int N = 64, K = 1024, M = 32, groupSize = 64;
  const int nGroups = K / groupSize;
  const int halfK = K / 2;
  int failures = 0;

  // ---- synthetic test data (fixed seed) ----
  std::mt19937 rng(0);
  std::normal_distribution<float> wDist(0.0f, 1.0f);
  std::normal_distribution<float> xDist(0.0f, 1.0f);

  std::vector<float> W(static_cast<size_t>(N) * K);
  for (auto& v : W) v = wDist(rng) * 0.05f;
  std::vector<float> X(static_cast<size_t>(M) * K);
  for (auto& v : X) v = xDist(rng) * 0.1f;

  // ---- Stage 1: fake quant ----
  std::vector<float> Wdq(W.size());
  std::vector<float> scaleFake(static_cast<size_t>(N) * nGroups);
  fakeQuantRtn(W.data(), N, K, groupSize, Wdq.data(), scaleFake.data());

  double sumAbsErr = 0.0;
  for (size_t i = 0; i < W.size(); ++i) sumAbsErr += std::fabs(Wdq[i] - W[i]);
  double meanAbsErr = sumAbsErr / W.size();
  bool check1 = meanAbsErr < 0.01;
  failures += !check1;
  std::printf("[1] fake-quant sanity: mean abs error = %.6f -> %s\n",
              meanAbsErr, check1 ? "PASS" : "FAIL");

  // ---- Stage 2: real quant ----
  std::vector<int8_t> Wint(W.size());
  std::vector<float> scaleReal(scaleFake.size());
  realQuantRtn(W.data(), N, K, groupSize, Wint.data(), scaleReal.data());

  bool check2 = true;
  int firstMismatch2 = -1;
  for (int n = 0; n < N && check2; ++n) {
    for (int g = 0; g < nGroups && check2; ++g) {
      float s = scaleFake[n * nGroups + g];
      int k0 = g * groupSize, k1 = k0 + groupSize;
      for (int k = k0; k < k1; ++k) {
        int idx = n * K + k;
        float qFromStage1 = std::round(Wdq[idx] / s);
        if (static_cast<int8_t>(qFromStage1) != Wint[idx]) {
          check2 = false;
          firstMismatch2 = idx;
          break;
        }
      }
    }
  }
  failures += !check2;
  std::printf("[2] real-quant vs fake-quant equivalence -> %s%s\n",
              check2 ? "PASS" : "FAIL",
              check2 ? "" : (" (first mismatch at flat index " +
                              std::to_string(firstMismatch2) + ")")
                                 .c_str());

  // ---- Stage 3: pack / unpack round trip ----
  std::vector<uint8_t> Wpacked(static_cast<size_t>(N) * halfK);
  packW4(Wint.data(), N, K, Wpacked.data());
  std::vector<int8_t> WintRoundTrip(Wint.size());
  unpackW4(Wpacked.data(), N, K, WintRoundTrip.data());

  bool check3 = true;
  int firstMismatch3 = -1;
  for (size_t i = 0; i < Wint.size(); ++i) {
    if (Wint[i] != WintRoundTrip[i]) {
      check3 = false;
      firstMismatch3 = static_cast<int>(i);
      break;
    }
  }
  failures += !check3;
  std::printf("[3] pack/unpack round-trip -> %s%s\n",
              check3 ? "PASS" : "FAIL",
              check3 ? "" : (" (first mismatch at flat index " +
                              std::to_string(firstMismatch3) + ")")
                                 .c_str());

  // ---- runtime activation quant ----
  std::vector<int8_t> Xint8(X.size());
  std::vector<float> xScale(M);
  quantizeActivationInt8(X.data(), M, K, Xint8.data(), xScale.data());

  // ---- CPU reference matmul (fp32 ground truth) ----
  std::vector<float> Yref(static_cast<size_t>(M) * N);
  cpuReferenceMatmul(X.data(), W.data(), M, N, K, Yref.data());

  // ---- CPU simulation of the kernel algorithm ----
  std::vector<float> Ycpu(Yref.size());
  cpuW4A8Simulate(Xint8.data(), xScale.data(), Wpacked.data(),
                   scaleReal.data(), M, N, K, groupSize, Ycpu.data());

  // ---- GPU kernel ----
  int8_t* dXint8 = nullptr;
  float* dXScale = nullptr;
  uint8_t* dWpacked = nullptr;
  float* dWScale = nullptr;
  float* dY = nullptr;
  CUDA_CHECK(cudaMalloc(&dXint8, Xint8.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&dXScale, xScale.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dWpacked, Wpacked.size() * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&dWScale, scaleReal.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY, Yref.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dXint8, Xint8.data(), Xint8.size() * sizeof(int8_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dXScale, xScale.data(), xScale.size() * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWpacked, Wpacked.data(),
                         Wpacked.size() * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWScale, scaleReal.data(),
                         scaleReal.size() * sizeof(float),
                         cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  w4a8GemmNaiveKernel<<<grid, block>>>(dXint8, dXScale, dWpacked, dWScale, dY,
                                        M, N, K, groupSize, nGroups);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> Ygpu(Yref.size());
  CUDA_CHECK(cudaMemcpy(Ygpu.data(), dY, Ygpu.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(dXint8));
  CUDA_CHECK(cudaFree(dXScale));
  CUDA_CHECK(cudaFree(dWpacked));
  CUDA_CHECK(cudaFree(dWScale));
  CUDA_CHECK(cudaFree(dY));

  // ---- Check 4: CPU-sim vs GPU-kernel (isolates real kernel bugs) ----
  float maxAbsDiff = 0.0f;
  for (size_t i = 0; i < Ycpu.size(); ++i)
    maxAbsDiff = std::max(maxAbsDiff, std::fabs(Ycpu[i] - Ygpu[i]));
  bool check4 = maxAbsDiff < 1e-5f;
  failures += !check4;
  std::printf("[4] CPU-sim vs GPU-kernel: max abs diff = %.8f -> %s\n",
              maxAbsDiff, check4 ? "PASS" : "FAIL");

  // ---- Check 5: GPU kernel vs fp32 ground truth (sanity bound, not a hard gate) ----
  double sumAbsDiff = 0.0, sumAbsRef = 0.0;
  for (size_t i = 0; i < Yref.size(); ++i) {
    sumAbsDiff += std::fabs(Ygpu[i] - Yref[i]);
    sumAbsRef += std::fabs(Yref[i]);
  }
  double meanAbsDiff = sumAbsDiff / Yref.size();
  double meanAbsRef = sumAbsRef / Yref.size();
  double meanRelErr = meanAbsDiff / meanAbsRef;
  const char* band = meanRelErr < 0.20 ? "expected range for uncalibrated RTN"
                     : meanRelErr < 0.30
                         ? "higher than usual, but not necessarily a bug"
                         : "unexpectedly high, worth investigating";
  std::printf(
      "[5] GPU kernel vs fp32 ground truth: mean rel error = %.4f (%.2f%%) "
      "-> %s [sanity bound, not a strict pass/fail]\n",
      meanRelErr, meanRelErr * 100.0, band);

  std::printf("\n%d/4 strict checks failed (check 5 is informational only)\n",
              failures);
  return failures == 0 ? 0 : 1;
}
