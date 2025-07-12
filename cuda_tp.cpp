#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdint.h>

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    float *__restrict__ cb_palette, uint32_t *__restrict__ block_jobs,
    uint32_t *__restrict__ block_job_sizes, size_t in1_size, size_t in2_size,
    size_t cb_palette_size, size_t out_size, int batch_size);

torch::Tensor tensor_product_forward(torch::Tensor in1, torch::Tensor in2,
                                     torch::Tensor out,
                                     torch::Tensor cb_palette,
                                     torch::Tensor block_jobs,
                                     torch::Tensor block_job_sizes) {

  tensor_product_forward_cuda(
      in1.data<float>(), in2.data<float>(), out.data<float>(),
      cb_palette.data<float>(), block_jobs.data<uint32_t>(),
      block_job_sizes.data<uint32_t>(), in1.size(1), in2.size(1),
      cb_palette.size(0), out.size(1), in1.size(0));

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tensor_product_forward, "Tensor Product forward (CUDA)");
}
