#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_THREADS 16
#define NUM_BLOCKS 1024
#define MINIBATCH_MAX_SIZE NUM_BLOCKS
#define CHUNKS 1

__global__ void tensor_product_forward_kernel(
    float *__restrict__ in1_global, float *__restrict__ in2_global,
    float *__restrict__ out, uint32_t *__restrict__ out_indices,
    float *__restrict__ cb_lut_global, uint32_t *__restrict__ input_indices,
    size_t len_indices, size_t in1_size, size_t in2_size, size_t cb_lut_size,
    size_t out_size, size_t minibatch_size) {
  int batch_idx = blockIdx.x;
  in1_global += batch_idx * in1_size;
  in2_global += batch_idx * in2_size;
  out += batch_idx * out_size;

  extern __shared__ float shared_mem[];
  float *in1 = shared_mem;
  float *in2 = shared_mem + in1_size;
  float *cb_lut = shared_mem + in1_size + in2_size;

  for (int i = threadIdx.x; i < in1_size; i += blockDim.x) {
    in1[i] = in1_global[i];
  }
  for (int i = threadIdx.x; i < in2_size; i += blockDim.x) {
    in2[i] = in2_global[i];
  }
  for (int i = threadIdx.x; i < cb_lut_size; i += blockDim.x) {
    cb_lut[i] = cb_lut_global[i];
  }
  __syncthreads();

  int curr_len_indices = (len_indices + (NUM_THREADS - 1)) / NUM_THREADS;
  int start_idx = curr_len_indices * threadIdx.x;
  int end_idx = start_idx + curr_len_indices;
  end_idx = end_idx < len_indices ? end_idx : len_indices;
  if (batch_idx >= minibatch_size || start_idx >= len_indices) {
    return;
  }

  float acc = 0.0;
  int i = start_idx;
  int out_index = out_indices[i];
  acc += in1[(input_indices[i] >> 10) & 0x3FF] *
         in2[input_indices[i] & 0x3FF] *
         cb_lut[(input_indices[i] >> 20) & 0x3FF];
  int flush_acc = input_indices[i] >> 31;
  if (flush_acc) {
    atomicAdd(out + out_index, acc);
  }
  out_index += flush_acc;
  acc *= (float)(flush_acc ^ 0x1);
  i++;
#pragma unroll
  for (i; i < end_idx; i++) {
    uint32_t input_idx = input_indices[i];
    acc += in1[(input_idx >> 10) & 0x3FF] *
           in2[input_idx & 0x3FF] *
           cb_lut[(input_idx >> 20) & 0x3FF];
    flush_acc = input_idx >> 31;
    if (flush_acc) {
      out[out_index] = acc;
    }
    out_index += flush_acc;
    acc *= (float)(flush_acc ^ 0x1);
  }
  atomicAdd(out + out_index, acc);
}

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint32_t *__restrict__ out_indices, float *__restrict__ cb_lut,
    uint32_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t cb_lut_size, size_t out_size, int batch_size) {
  while (batch_size > 0) {
    for (int i = 0; i < CHUNKS; i++) {
      int minibatch_size = batch_size < MINIBATCH_MAX_SIZE ? batch_size : MINIBATCH_MAX_SIZE;
      tensor_product_forward_kernel<<<NUM_BLOCKS, NUM_THREADS,
                                      (in1_size + in2_size + cb_lut_size) *
                                          sizeof(float)>>>(
          in1, in2, out, out_indices, cb_lut, input_indices,
          len_indices / CHUNKS, in1_size, in2_size, cb_lut_size, out_size,
          minibatch_size);
      out_indices += len_indices / CHUNKS;
      input_indices += len_indices / CHUNKS;
    }
    out_indices -= len_indices;
    input_indices -= len_indices;
    batch_size -= MINIBATCH_MAX_SIZE;
    in1 += in1_size * MINIBATCH_MAX_SIZE;
    in2 += in2_size * MINIBATCH_MAX_SIZE;
    out += out_size * MINIBATCH_MAX_SIZE;
  }
}
