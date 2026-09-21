// pybind binding, kept out of the .cu: torch/extension.h and CUDA's cuda::std headers
// together trigger "error C2872: 'std' ambiguous" under MSVC.

#include <torch/extension.h>

#include <tuple>

std::tuple<at::Tensor, at::Tensor> fake_quant_group(
    const at::Tensor& x, int64_t num_bits, int64_t group_size, bool return_scale);

at::Tensor wq_gemv(
    const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& wscale, int64_t out_group);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fake_quant_group",
      &fake_quant_group,
      "Fused symmetric RTN quant-dequant, grouped along the last axis (CUDA)",
      py::arg("x"),
      py::arg("num_bits"),
      py::arg("group_size"),
      py::arg("return_scale") = false);
  m.def(
      "wq_gemv",
      &wq_gemv,
      "Weight-only quantized matmul: int8 weight, bf16 activation (CUDA)",
      py::arg("x"),
      py::arg("qweight"),
      py::arg("wscale"),
      py::arg("out_group") = 0);
}
