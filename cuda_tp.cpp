#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdint.h>

void tensor_product_forward_cuda(
  float* in1,
  float* in2,
  float* out,
  uint16_t* __restrict__ in1_indices,
  uint16_t* __restrict__ in2_indices,
  uint16_t* __restrict__ out_indices,
  float* cb_lut,
  uint16_t* cb_indices,
  size_t len_indices,
  size_t in1_size,
  size_t in2_size,
  size_t out_size,
  int batch_size);

torch::Tensor tensor_product_forward(
  torch::Tensor in1,
  torch::Tensor in2,
  torch::Tensor out,
  torch::Tensor in1_indices,
  torch::Tensor in2_indices,
  torch::Tensor out_indices,
  torch::Tensor cb_lut,
  torch::Tensor cb_indices) {

  tensor_product_forward_cuda(
    in1.data<float>(),
    in2.data<float>(),
    out.data<float>(),
    in1_indices.data<uint16_t>(),
    in2_indices.data<uint16_t>(),
    out_indices.data<uint16_t>(),
    cb_lut.data<float>(),
    cb_indices.data<uint16_t>(),
    in1_indices.size(0),
    in1.size(1),
    in2.size(1),
    out.size(1),
    in1.size(0));

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tensor_product_forward, "Tensor Product forward (CUDA)");
}
