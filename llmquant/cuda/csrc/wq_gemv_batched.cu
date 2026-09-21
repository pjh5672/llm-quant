// Weight-only quantized GEMV across many experts, in one launch.
//
// A Mixture-of-Experts decode step runs the same tiny matmul once per expert the token was
// routed to: top-8 of 64, two projections, sixteen layers -- 256 separate launches for a
// single token. Measured on OLMoE-1B-7B, the expert block took 4.01 ms against bf16's
// 0.955 ms while the matmuls inside it accounted for about 0.19 ms. The arithmetic was
// never the problem; doing it 256 times from Python was.
//
// So the rows are gathered first, grouped by expert, and the whole step becomes two
// launches. Each block still owns one output column and keeps its slice of the weight in
// registers across the rows that share it, exactly as wq_gemv does -- what changes is that
// the weight pointer is chosen per block from a stacked [E, N, groups, group_size] tensor.
//
// Blocks are laid out by the host rather than derived from blockIdx, because a block that
// straddled two experts would have to reload the weight halfway through and lose the reuse
// that makes this worth doing. The host knows the expert boundaries already -- it just
// grouped the rows by them -- so it emits (expert, first row, row count) per block.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kWarps = 4;
constexpr int kBlock = kWarps * 32;
constexpr int kPerLane = 4;
constexpr int kMaxRows = 8;

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <typename scalar_t>
__global__ void wq_gemv_batched_kernel(
    const scalar_t* __restrict__ x,        // [R, groups * group_size], grouped by expert
    const int8_t* __restrict__ qweight,    // [E, N, groups, group_size]
    const float* __restrict__ wscale,      // [E, scale_rows, groups]
    const int* __restrict__ block_expert,  // [B]
    const int* __restrict__ block_row0,    // [B]
    const int* __restrict__ block_rows,    // [B]
    scalar_t* __restrict__ out,            // [R, N]
    const int N,
    const int groups,
    const int group_size,
    const int out_group,
    const int scale_rows) {
  __shared__ float smem[kWarps][kMaxRows];

  const int n = blockIdx.x;
  const int b = blockIdx.y;
  if (n >= N) {
    return;
  }
  const int expert = block_expert[b];
  const int row0 = block_row0[b];
  const int rows = block_rows[b];
  if (rows <= 0) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int scale_row = out_group > 0 ? n / out_group : n;
  const int k_total = groups * group_size;
  const int lanes_per_group = group_size / kPerLane;

  // the only difference from the single-expert kernel: where this block's weight lives
  const int8_t* qw = qweight + static_cast<int64_t>(expert) * N * k_total;
  const float* ws = wscale + static_cast<int64_t>(expert) * scale_rows * groups;

  float totals[kMaxRows];
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r) {
    totals[r] = 0.0f;
  }

  for (int g = warp; g < groups; g += kWarps) {
    const int64_t base = static_cast<int64_t>(g) * group_size + lane * kPerLane;
    int8_t w[kPerLane];
    if (lane < lanes_per_group) {
      const int32_t packed =
          *reinterpret_cast<const int32_t*>(qw + static_cast<int64_t>(n) * k_total + base);
      *reinterpret_cast<int32_t*>(w) = packed;
    } else {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        w[i] = 0;
      }
    }
    const float scale = ws[static_cast<int64_t>(scale_row) * groups + g];

    for (int r = 0; r < rows; ++r) {
      float partial = 0.0f;
      if (lane < lanes_per_group) {
        const scalar_t* row = x + static_cast<int64_t>(row0 + r) * k_total + base;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          partial += static_cast<float>(row[i]) * static_cast<float>(w[i]);
        }
      }
      // same ordering as wq_gemv: scale the lane's share, reduce once at the end. Keeping
      // it identical is what makes the batched path bit-exact against the looped one.
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
    out[static_cast<int64_t>(row0 + r) * N + n] = static_cast<scalar_t>(total);
  }
}

}  // namespace

at::Tensor wq_gemv_batched(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const at::Tensor& wscale,
    const at::Tensor& block_expert,
    const at::Tensor& block_row0,
    const at::Tensor& block_rows,
    int64_t out_group) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && wscale.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(qweight.dim() == 4, "qweight must be [E, N, groups, group_size]");
  TORCH_CHECK(wscale.dim() == 3, "wscale must be [E, scale_rows, groups]");
  TORCH_CHECK(qweight.scalar_type() == at::kChar, "qweight must be int8");
  TORCH_CHECK(wscale.scalar_type() == at::kFloat, "wscale must be float32");

  const at::cuda::CUDAGuard guard(x.device());
  const auto xc = x.contiguous();
  const auto qw = qweight.contiguous();
  const auto ws = wscale.contiguous();
  const auto be = block_expert.to(at::kInt).contiguous();
  const auto br0 = block_row0.to(at::kInt).contiguous();
  const auto brn = block_rows.to(at::kInt).contiguous();

  const int64_t R = xc.size(0);
  const int64_t N = qw.size(1);
  const int64_t groups = qw.size(2);
  const int64_t group_size = qw.size(3);
  const int64_t scale_rows = ws.size(1);
  const int64_t blocks = be.size(0);

  TORCH_CHECK(xc.size(1) == groups * group_size, "x width must be groups * group_size");
  TORCH_CHECK(group_size % kPerLane == 0 && group_size / kPerLane <= 32,
              "group_size must be a multiple of ", kPerLane, " and at most ", kPerLane * 32);

  auto out = at::empty({R, N}, xc.options());
  if (blocks == 0 || R == 0) {
    return out;
  }

  const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(blocks));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(xc.scalar_type(), "wq_gemv_batched", [&] {
    wq_gemv_batched_kernel<scalar_t><<<grid, kBlock, 0, stream>>>(
        xc.data_ptr<scalar_t>(), qw.data_ptr<int8_t>(), ws.data_ptr<float>(),
        be.data_ptr<int>(), br0.data_ptr<int>(), brn.data_ptr<int>(),
        out.data_ptr<scalar_t>(), static_cast<int>(N), static_cast<int>(groups),
        static_cast<int>(group_size), static_cast<int>(out_group),
        static_cast<int>(scale_rows));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t wq_gemv_batched_max_rows() { return kMaxRows; }
