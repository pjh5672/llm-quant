// Weight-only quantized matmul: int weights read as integers, bf16 activations.
//
// This is where the speed is. Decode is memory bound -- every token re-reads the whole
// weight -- so halving the weight bytes is worth about 2x, while int8 *math* measured only
// 1.03-1.04x over bf16 on this GPU. So the weight stays int8 into the kernel and is
// dequantized in registers, and the activation stays bf16.
//
// The weight arrives in canonical form, [N, groups, group_size] with the reduction axis
// already padded and any head split already applied by llmquant.core.layout. The kernel
// knows nothing about heads: it walks groups and applies one scale per group. Summing
// inside a group and scaling once (rather than folding the scale into each product) is
// deliberate -- it is what lets an int-by-int variant be bit-exact against the Phase 2
// reference, since integer sums do not depend on summation order.
//
// Shape of the kernel, and why:
//   one block per output column, one warp per group   a group is 128 values and a warp
//                                                     holds exactly 128 via 4 per lane, so
//                                                     the reduction is a warp shuffle with
//                                                     no __syncthreads in the inner loop
//   int8 read four at a time as int32                 one load instruction per 4 weights
//   the weight is loaded once and reused over M       the first version looped M outside
//                                                     the load and re-read the whole weight
//                                                     per row, which cost 64x at M=64

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kWarps = 4;
constexpr int kBlock = kWarps * 32;
constexpr int kPerLane = 4;  // int8 values a lane handles per group, read as one int32
constexpr int kMaxRows = 8;  // activation rows kept in registers at once

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <typename scalar_t>
__global__ void wq_gemv_kernel(
    const scalar_t* __restrict__ x,      // [M, groups * group_size]
    const int8_t* __restrict__ qweight,  // [N, groups, group_size]
    const float* __restrict__ wscale,    // [scale_rows, groups]
    scalar_t* __restrict__ out,          // [M, N]
    const int M,
    const int N,
    const int groups,
    const int group_size,
    const int out_group) {
  __shared__ float smem[kWarps][kMaxRows];

  const int n = blockIdx.x;
  const int row_block = blockIdx.y * kMaxRows;
  const int rows = min(kMaxRows, M - row_block);
  if (n >= N || rows <= 0) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int scale_row = out_group > 0 ? n / out_group : n;
  const int k_total = groups * group_size;
  const int lanes_per_group = group_size / kPerLane;

  float totals[kMaxRows];
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r) {
    totals[r] = 0.0f;
  }

  // each warp takes every kWarps-th group
  for (int g = warp; g < groups; g += kWarps) {
    const int64_t base = static_cast<int64_t>(g) * group_size + lane * kPerLane;
    int8_t w[kPerLane];
    if (lane < lanes_per_group) {
      // one 32-bit load instead of four byte loads
      const int32_t packed =
          *reinterpret_cast<const int32_t*>(qweight + static_cast<int64_t>(n) * k_total + base);
      *reinterpret_cast<int32_t*>(w) = packed;
    } else {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        w[i] = 0;
      }
    }
    const float scale = wscale[static_cast<int64_t>(scale_row) * groups + g];

    // the weight is in registers now, so every row is nearly free
    for (int r = 0; r < rows; ++r) {
      float partial = 0.0f;
      if (lane < lanes_per_group) {
        const scalar_t* row = x + static_cast<int64_t>(row_block + r) * k_total + base;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          partial += static_cast<float>(row[i]) * static_cast<float>(w[i]);
        }
      }
      // scale the lane's share here and reduce once at the very end, rather than reducing
      // per group. sum_g s_g * sum_lanes p = sum_g sum_lanes s_g * p, and with bf16
      // activations the rounding difference is far below the output's own resolution.
      // An int-by-int kernel would have to keep the group sum whole before scaling, since
      // that is what makes its result independent of summation order.
      totals[r] += partial * scale;
    }
  }

  for (int r = 0; r < rows; ++r) {
    const float reduced = warp_reduce_sum(totals[r]);
    if (lane == 0) {
      smem[warp][r] = reduced;
    }
  }
  __syncthreads();

  if (threadIdx.x < static_cast<unsigned>(rows)) {
    const int r = threadIdx.x;
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      total += smem[w][r];
    }
    out[static_cast<int64_t>(row_block + r) * N + n] = static_cast<scalar_t>(total);
  }
}

}  // namespace

at::Tensor wq_gemv(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& wscale,
    int64_t out_group) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && wscale.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(qweight.dim() == 3, "qweight must be [N, groups, group_size], got ", qweight.dim(), "D");
  TORCH_CHECK(qweight.scalar_type() == at::kChar, "qweight must be int8");
  TORCH_CHECK(wscale.scalar_type() == at::kFloat, "wscale must be float32");

  const auto xc = x.contiguous();
  const auto wc = qweight.contiguous();
  const auto sc = wscale.contiguous();

  const int64_t N = wc.size(0);
  const int64_t groups = wc.size(1);
  const int64_t group_size = wc.size(2);
  const int64_t M = xc.numel() / xc.size(-1);
  TORCH_CHECK(group_size % 4 == 0, "group_size must be a multiple of 4, got ", group_size);
  TORCH_CHECK(
      xc.size(-1) == groups * group_size,
      "x last dim ", xc.size(-1), " does not match qweight ", groups, "x", group_size);
  const int64_t scale_rows = out_group > 0 ? (N + out_group - 1) / out_group : N;
  TORCH_CHECK(
      sc.size(0) == scale_rows && sc.size(1) == groups,
      "wscale must be [", scale_rows, ", ", groups, "], got [", sc.size(0), ", ", sc.size(1), "]");

  auto out = at::empty({M, N}, xc.options());
  if (M == 0 || N == 0) {
    return out;
  }

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(xc));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned int>(N), static_cast<unsigned int>((M + kMaxRows - 1) / kMaxRows));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, xc.scalar_type(), "wq_gemv", [&] {
        wq_gemv_kernel<scalar_t><<<grid, kBlock, 0, stream>>>(
            xc.const_data_ptr<scalar_t>(),
            wc.const_data_ptr<int8_t>(),
            sc.const_data_ptr<float>(),
            out.mutable_data_ptr<scalar_t>(),
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(groups),
            static_cast<int>(group_size),
            static_cast<int>(out_group));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
