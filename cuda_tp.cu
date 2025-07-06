#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_THREADS 128
#define NUM_BLOCKS 1024
#define MINIBATCH_MAX_SIZE NUM_BLOCKS

__global__ void tensor_product_forward_kernel(
    float *__restrict__ in1_global, float *__restrict__ in2_global, float *__restrict__ out,
    uint32_t *__restrict__ out_indices, float *__restrict__ cb_palette_global,
    uint64_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t cb_palette_size, size_t out_size, size_t minibatch_size) {
  int batch_idx = blockIdx.x;
  in1_global += batch_idx * in1_size;
  in2_global += batch_idx * in2_size;
  out += batch_idx * out_size;

  // Copy in1, in2, and the Clebsch-Gordon coefficient palette to shared memory.
  extern __shared__ float shared_mem[];
  float* in1 = shared_mem;
  float* in2 = shared_mem + in1_size;
  float* cb_palette = shared_mem + in1_size + in2_size;
  for (int i = threadIdx.x; i < in1_size; i += blockDim.x) {
    in1[i] = in1_global[i];
  }
  for (int i = threadIdx.x; i < in2_size; i += blockDim.x) {
    in2[i] = in2_global[i];
  }
  for (int i = threadIdx.x; i < cb_palette_size; i += blockDim.x) {
    cb_palette[i] = cb_palette_global[i];
  }
  __syncthreads();

  if (batch_idx >= minibatch_size || threadIdx.x >= len_indices) {
    return;
  }

  // Perform the actual tensor product.
#pragma unroll
  for (int i = threadIdx.x; i < len_indices; i += blockDim.x) {
    // Decompress our input and Clebsch-Gordon indices.
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

    // Perform the actual mul-adds.
    float acc = in1[in1_idx1] * in2[in2_idx1] * cb_palette[cb_idx1];
    acc += in1[in1_idx2] * in2[in2_idx2] * cb_palette[cb_idx2];
    acc += in1[in1_idx3] * in2[in2_idx3] * cb_palette[cb_idx3];

    // Write back out to our output.
    atomicAdd(out + out_indices[i], acc);
  }
}

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint32_t *__restrict__ out_indices, float *__restrict__ cb_palette,
    uint64_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t cb_palette_size, size_t out_size, int batch_size) {
  while (batch_size > 0) {
      int minibatch_size = batch_size < MINIBATCH_MAX_SIZE ? batch_size : MINIBATCH_MAX_SIZE;
      tensor_product_forward_kernel<<<NUM_BLOCKS, NUM_THREADS, (in1_size + in2_size + cb_palette_size) * sizeof(float)>>>(
          in1, in2, out, out_indices, cb_palette, input_indices, len_indices,
	  in1_size, in2_size, cb_palette_size, out_size, minibatch_size);
      out_indices += len_indices;
      input_indices += len_indices;
    out_indices -= len_indices;
    input_indices -= len_indices;
    batch_size -= MINIBATCH_MAX_SIZE;
    in1 += in1_size * MINIBATCH_MAX_SIZE;
    in2 += in2_size * MINIBATCH_MAX_SIZE;
    out += out_size * MINIBATCH_MAX_SIZE;
  }
}
