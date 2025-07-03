#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 16
#define NUM_BLOCKS 256
#define MINIBATCH_MAX_SIZE NUM_THREADS

__global__ void tensor_product_forward_kernel(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint16_t *__restrict__ out_indices, float *__restrict__ cb_lut,
    uint32_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t out_size, size_t minibatch_size) {
  int curr_len_indices = (len_indices + NUM_BLOCKS - 1) / NUM_BLOCKS;
  int start_idx = curr_len_indices * blockIdx.x;
  int end_idx = start_idx + curr_len_indices;
  end_idx = end_idx < len_indices ? end_idx : len_indices;
  if (threadIdx.x >= minibatch_size) {
    return;
  }

  in1 += threadIdx.x * in1_size;
  in2 += threadIdx.x * in2_size;
  out += threadIdx.x * out_size;
  float acc = 0.0;
  int i = start_idx;
  int out_index = out_indices[i];
  for (start_idx; i < end_idx; i++) {
    acc += in1[(input_indices[i] >> 10) & 0x3FF] *
           in2[input_indices[i] & 0x3FF] *
           cb_lut[(input_indices[i] >> 20) & 0x3FF];
    if (input_indices[i] & 0x80000000) {
      out[out_index] += acc;
      acc = 0.0;
      out_index++;
    }
  }
  out[out_index] += acc;
}

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint16_t *__restrict__ out_indices, float *__restrict__ cb_lut,
    uint32_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t out_size, int batch_size) {
  while (batch_size > 0) {
    int minibatch_size = batch_size < MINIBATCH_MAX_SIZE ? batch_size : MINIBATCH_MAX_SIZE;
    tensor_product_forward_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
        in1, in2, out, out_indices, cb_lut, input_indices, len_indices,
        in1_size, in2_size, out_size, minibatch_size);
    batch_size -= MINIBATCH_MAX_SIZE;
    in1 += in1_size * MINIBATCH_MAX_SIZE;
    in2 += in2_size * MINIBATCH_MAX_SIZE;
    out += out_size * MINIBATCH_MAX_SIZE;
  }
}
