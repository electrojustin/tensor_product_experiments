#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdint.h>

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint32_t *__restrict__ out_indices, float *__restrict__ cb_lut,
    uint64_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t out_size, int batch_size);

torch::Tensor tensor_product_forward(torch::Tensor in1, torch::Tensor in2,
                                     torch::Tensor out,
                                     torch::Tensor out_indices,
                                     torch::Tensor cb_lut,
                                     torch::Tensor input_indices) {

  tensor_product_forward_cuda(
      in1.data<float>(), in2.data<float>(), out.data<float>(),
      out_indices.data<uint32_t>(), cb_lut.data<float>(),
      input_indices.data<uint64_t>(), input_indices.size(0), in1.size(1),
      in2.size(1), out.size(1), in1.size(0));

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tensor_product_forward, "Tensor Product forward (CUDA)");
}
