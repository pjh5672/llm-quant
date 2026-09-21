// Fused symmetric RTN quant-dequant (fake quant), grouped along the last (reduction) axis.
//
// Must stay BIT-EXACT with llmquant.utils.quant_ops.fake_quantize:
//   scale = max(abs_max(group) / qmax, 1e-8)            fp32
//   q     = clamp(round_half_to_even(x / scale), qmin, qmax)
//   out   = cast_to_input_dtype(q * scale)              round-to-nearest-even
// So: fp32 math throughout, rintf (NOT roundf, which is half-away-from-zero), and no
// --use_fast_math (it would swap in an approximate reciprocal for the division).
//
// torch/extension.h is deliberately NOT included here: it pulls in compiled_autograd.h,
// which clashes with CUDA's cuda::std and fails on MSVC with "error C2872: 'std' ambiguous".
// The pybind binding lives in fake_quant.cpp instead.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <tuple>

namespace {

constexpr int kBlock = 128;   // one block per group; matches the default group_size
constexpr int kWarps = kBlock / 32;
constexpr float kScaleEps = 1e-8f;  // llmquant.observers.minmax.SCALE_EPS

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = smem[0];
#pragma unroll
    for (int i = 1; i < kWarps; ++i) {
      m = fmaxf(m, smem[i]);
    }
    smem[0] = m;
  }
  __syncthreads();
  return smem[0];
}

template <typename scalar_t>
__global__ void fake_quant_group_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    float* __restrict__ scale_out,  // nullable, one fp32 scale per group
    const int group_size,
    const float qmin,
    const float qmax) {
  __shared__ float smem[kWarps];
  __shared__ float s_scale;

  const int64_t base = static_cast<int64_t>(blockIdx.x) * group_size;

  float amax = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += kBlock) {
    amax = fmaxf(amax, fabsf(static_cast<float>(x[base + i])));
  }
  amax = block_reduce_max(amax, smem);

  if (threadIdx.x == 0) {
    // fmaxf matches Tensor.clamp(min=...)
    s_scale = fmaxf(amax / qmax, kScaleEps);
    if (scale_out != nullptr) {
      scale_out[blockIdx.x] = s_scale;
    }
  }
  __syncthreads();

  const float scale = s_scale;
  for (int i = threadIdx.x; i < group_size; i += kBlock) {
    const float v = static_cast<float>(x[base + i]);
    const float q = fminf(fmaxf(rintf(v / scale), qmin), qmax);
    out[base + i] = static_cast<scalar_t>(q * scale);
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> fake_quant_group(
    const at::Tensor& x,
    int64_t num_bits,
    int64_t group_size,
    bool return_scale) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  TORCH_CHECK(num_bits >= 2 && num_bits <= 8, "num_bits must be in [2, 8], got ", num_bits);
  TORCH_CHECK(group_size > 0, "group_size must be positive, got ", group_size);
  const auto dtype = x.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
      "x must be float32, float16 or bfloat16, got ", dtype);

  const auto xc = x.contiguous();
  const int64_t k = xc.size(-1);
  TORCH_CHECK(k % group_size == 0, "last dim ", k, " is not divisible by group_size ", group_size);

  auto out = at::empty_like(xc);
  const auto scale_opts = xc.options().dtype(at::kFloat);
  if (xc.numel() == 0) {
    return std::make_tuple(out.reshape(x.sizes()), at::empty({0}, scale_opts));
  }

  auto scale_sizes = xc.sizes().vec();
  scale_sizes.back() = k / group_size;
  auto scale = return_scale ? at::empty(scale_sizes, scale_opts) : at::empty({0}, scale_opts);

  const int64_t groups = xc.numel() / group_size;
  TORCH_CHECK(groups <= 2147483647LL, "too many groups for a 1D grid: ", groups);

  const float qmax = static_cast<float>((1 << (num_bits - 1)) - 1);
  const float qmin = -qmax - 1.0f;

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(xc));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, dtype, "fake_quant_group", [&] {
        fake_quant_group_kernel<scalar_t>
            <<<static_cast<unsigned int>(groups), kBlock, 0, stream>>>(
                xc.const_data_ptr<scalar_t>(),
                out.mutable_data_ptr<scalar_t>(),
                return_scale ? scale.mutable_data_ptr<float>() : nullptr,
                static_cast<int>(group_size),
                qmin,
                qmax);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(out.reshape(x.sizes()), scale);
}
