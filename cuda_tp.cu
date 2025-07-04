#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 16
#define NUM_BLOCKS 256
#define MINIBATCH_MAX_SIZE NUM_THREADS

__global__ void tensor_product_forward_kernel(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint32_t *__restrict__ out_indices, float *__restrict__ cb_lut,
    uint64_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t out_size, size_t minibatch_size) {
  int curr_len_indices = (len_indices + NUM_BLOCKS - 1) / NUM_BLOCKS;
  int start_idx = curr_len_indices * blockIdx.x;
  if (threadIdx.x >= minibatch_size) {
    return;
  }

  int end_idx = start_idx + curr_len_indices;
  end_idx = end_idx < len_indices ? end_idx : len_indices;
  in1 += threadIdx.x * in1_size;
  in2 += threadIdx.x * in2_size;
  out += threadIdx.x * out_size;
  float acc = 0.0;
  int i = start_idx;
  int out_index = out_indices[i];
  for (start_idx; i < end_idx; i++) {
    uint64_t input_idx = input_indices[i];
    int in1_idx1 = input_idx & 0x3FF;
    int in2_idx1 = (input_idx >> 10) & 0x3FF;
    int cb_idx1 = (input_idx >> 20) & 0x3FF;
    int in1_idx2 = ((input_idx >> 32) & 0x1F) ^ in1_idx1;
    int in2_idx2 = ((input_idx >> 37) & 0x1) + in2_idx1;
    int cb_idx2 = (input_idx >> 38) & 0x3FF;
    int in1_idx3 = ((input_idx >> 48) & 0x1F) ^ in1_idx1;
    int in2_idx3 = ((input_idx >> 53) & 0x1) + in2_idx2;
    int cb_idx3 = input_idx >> 54;
    acc += in1[in1_idx1] * in2[in2_idx1] * cb_lut[cb_idx1];
    acc += in1[in1_idx2] * in2[in2_idx2] * cb_lut[cb_idx2];
    acc += in1[in1_idx3] * in2[in2_idx3] * cb_lut[cb_idx3];
    if (input_indices[i] & 0x80000000) {
      atomicAdd(out + out_index, acc);
      acc = 0.0;
      out_index++;
    }
  }
  atomicAdd(out + out_index, acc);
}

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint32_t *__restrict__ out_indices, float *__restrict__ cb_lut,
    uint64_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
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
