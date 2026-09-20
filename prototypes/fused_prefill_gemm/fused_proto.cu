// Feasibility prototype: can a fused int8-weight GEMM reach cuBLAS-class prefill speed?
//
// The question is NOT whether fusing removes the dequant traffic -- it obviously does. It
// is whether a hand-written tensor-core GEMM can get close enough to cuBLAS that removing
// the traffic is a net win. Measured on this machine, prefill at M=512 runs the bf16 GEMMs
// at 43.5 TFLOPS against a ~47-50 TFLOPS peak, so cuBLAS is near the roof and a slow
// hand-written kernel would give the dequant saving straight back.
//
// So two kernels are built from one template, differing only in how the B tile is filled:
//   gemm_bf16   B is already bf16            -> measures how good this GEMM is at all
//   gemm_fused  B is int8 + fp32 group scale -> measures what fusing costs on top
//
// Shapes are our prefill case: C[M,N] = A[M,K] @ B[N,K]^T, both operands K-major, which is
// the layout mma.sync wants (a "TN" gemm) and needs no transpose anywhere.
//
// Deliberately simple: one tile config, padded shared memory instead of swizzling, no
// double buffering, no split-K. If this lands in the right neighbourhood the optimizations
// are known and incremental; if it lands at 30% of cuBLAS, hand-writing is the wrong path
// and CUTLASS is the answer.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace {

constexpr int kBM = 64;   // rows of A per block
constexpr int kBN = 64;   // rows of B (= columns of C) per block
constexpr int kBK = 32;   // reduction step
constexpr int kWarps = 4; // 2x2, each warp owns a 32x32 quadrant
constexpr int kThreads = kWarps * 32;
constexpr int kPad = 8;   // shared-memory row padding, in bf16 elements, to break bank conflicts
constexpr int kGroup = 128;

using bf16_t = __nv_bfloat16;

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// one m16n8k16 bf16 mma, fp32 accumulate
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const unsigned (&a)[4],
                                             const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const bf16_t* src) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(src)));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned (&r)[2], const bf16_t* src) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(src)));
}

// Fill the B tile in shared memory. This is the ONLY difference between the two kernels:
// the fused one reads int8 and applies the group scale on the way in, so the dequantized
// weight exists only in shared memory and never in DRAM.
template <bool kFused>
__device__ void load_b_tile(bf16_t* dst, const bf16_t* b16, const int8_t* qb,
                            const float* scale, int n0, int k0, int N, int K, int groups,
                            int tid) {
  constexpr int kStride = kBK + kPad;
  for (int idx = tid; idx < kBN * kBK; idx += kThreads) {
    const int r = idx / kBK;       // which output column
    const int c = idx % kBK;       // which k within the tile
    const int n = n0 + r;
    const int k = k0 + c;
    bf16_t v = __float2bfloat16(0.0f);
    if (n < N && k < K) {
      if (kFused) {
        // exactly the arithmetic the unfused path performs: fp32 scale, product rounded
        // once to bf16. The scale stays fp32 -- rounding it to bf16 would cost up to 49%
        // of the int8 quantization step.
        const float s = scale[static_cast<int64_t>(n) * groups + (k / kGroup)];
        v = __float2bfloat16(static_cast<float>(qb[static_cast<int64_t>(n) * K + k]) * s);
      } else {
        v = b16[static_cast<int64_t>(n) * K + k];
      }
    }
    dst[r * kStride + c] = v;
  }
}

template <bool kFused>
__global__ __launch_bounds__(kThreads) void gemm_kernel(
    const bf16_t* __restrict__ a, const bf16_t* __restrict__ b16,
    const int8_t* __restrict__ qb, const float* __restrict__ scale,
    bf16_t* __restrict__ c, int M, int N, int K, int groups) {
  constexpr int kStride = kBK + kPad;
  __shared__ bf16_t sa[kBM * kStride];
  __shared__ bf16_t sb[kBN * kStride];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int warp_m = (warp / 2) * 32;  // 2x2 warp grid over the 64x64 tile
  const int warp_n = (warp % 2) * 32;

  const int m0 = blockIdx.y * kBM;
  const int n0 = blockIdx.x * kBN;

  float acc[2][4][4];  // 2 m-tiles x 4 n-tiles of m16n8k16, 4 floats each
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kBK) {
    // A tile
    for (int idx = tid; idx < kBM * kBK; idx += kThreads) {
      const int r = idx / kBK, c = idx % kBK;
      const int m = m0 + r, k = k0 + c;
      sa[r * kStride + c] = (m < M && k < K) ? a[static_cast<int64_t>(m) * K + k]
                                             : __float2bfloat16(0.0f);
    }
    load_b_tile<kFused>(sb, b16, qb, scale, n0, k0, N, K, groups, tid);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      unsigned af[2][4], bf[4][2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        // ldmatrix m8n8.x4 wants each lane to point at the row it owns
        const int row = warp_m + i * 16 + (lane % 16);
        const int col = kk + (lane / 16) * 8;
        ldmatrix_x4(af[i], &sa[row * kStride + col]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int row = warp_n + j * 8 + (lane % 8);
        const int col = kk + (lane / 8) * 8;
        ldmatrix_x2(bf[j], &sb[row * kStride + col]);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) mma_m16n8k16(acc[i][j], af[i], bf[j]);
    }
    __syncthreads();
  }

  // m16n8k16 accumulator layout: lane L holds rows (L/4) and (L/4 + 8), cols (L%4)*2 + {0,1}
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = m0 + warp_m + i * 16 + (lane / 4) + (e / 2) * 8;
        const int n = n0 + warp_n + j * 8 + (lane % 4) * 2 + (e % 2);
        if (m < M && n < N) c[static_cast<int64_t>(m) * N + n] = __float2bfloat16(acc[i][j][e]);
      }
    }
  }
}

at::Tensor run(const at::Tensor& a, const at::Tensor& b16, const at::Tensor& qb,
               const at::Tensor& scale, bool fused) {
  const at::cuda::CUDAGuard guard(a.device());
  const int M = a.size(0), K = a.size(1);
  const int N = fused ? qb.size(0) : b16.size(0);
  const int groups = fused ? static_cast<int>(scale.size(1)) : 1;
  auto c = at::empty({M, N}, a.options());

  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* ap = reinterpret_cast<const bf16_t*>(a.data_ptr());
  auto* cp = reinterpret_cast<bf16_t*>(c.data_ptr());
  if (fused) {
    gemm_kernel<true><<<grid, kThreads, 0, stream>>>(
        ap, nullptr, qb.data_ptr<int8_t>(), scale.data_ptr<float>(), cp, M, N, K, groups);
  } else {
    gemm_kernel<false><<<grid, kThreads, 0, stream>>>(
        ap, reinterpret_cast<const bf16_t*>(b16.data_ptr()), nullptr, nullptr, cp, M, N, K,
        groups);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

}  // namespace

// Registration lives in the .cpp: including torch/extension.h from a .cu makes `std`
// ambiguous under MSVC, which is why the shipped kernels are split the same way.
at::Tensor fused_proto_run(const at::Tensor& a, const at::Tensor& b16, const at::Tensor& qb,
                           const at::Tensor& scale, bool fused) {
  return run(a, b16, qb, scale, fused);
}
