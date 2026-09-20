// pybind for the feasibility prototype; see fused_proto.cu for what it measures.
#include <torch/extension.h>

at::Tensor fused_proto_run(const at::Tensor& a, const at::Tensor& b16, const at::Tensor& qb,
                           const at::Tensor& scale, bool fused);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &fused_proto_run, "fused int8-weight gemm prototype");
}
