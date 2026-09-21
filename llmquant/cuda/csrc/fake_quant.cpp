// pybind binding, kept out of the .cu: torch/extension.h and CUDA's cuda::std headers
// together trigger "error C2872: 'std' ambiguous" under MSVC.

#include <torch/extension.h>

#include <tuple>

std::tuple<at::Tensor, at::Tensor> fake_quant_group(
    const at::Tensor& x, int64_t num_bits, int64_t group_size, bool return_scale);

at::Tensor wq_gemv(
    const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& wscale, int64_t out_group);

at::Tensor wq_gemv_batched(
    const at::Tensor& x, const at::Tensor& qweight, const at::Tensor& wscale,
    const at::Tensor& block_expert, const at::Tensor& block_row0,
    const at::Tensor& block_rows, int64_t out_group);

int64_t wq_gemv_batched_max_rows();

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
  m.def(
      "wq_gemv_batched",
      &wq_gemv_batched,
      "Weight-only quantized matmul across a stack of experts, one launch (CUDA)",
      py::arg("x"),
      py::arg("qweight"),
      py::arg("wscale"),
      py::arg("block_expert"),
      py::arg("block_row0"),
      py::arg("block_rows"),
      py::arg("out_group") = 0);
  m.def("wq_gemv_batched_max_rows", &wq_gemv_batched_max_rows,
        "Rows one batched block keeps in registers at once");
}
