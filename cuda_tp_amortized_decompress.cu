#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_THREADS 64
#define NUM_BLOCKS 512
#define SAMPLES_PER_BLOCK 16
#define MAX_GULP_SIZE 256

#define MINIBATCH_MAX_SIZE (NUM_BLOCKS * SAMPLES_PER_BLOCK)
#define ALIGN_UP(X, ALIGNMENT) (ALIGNMENT * ((X + ALIGNMENT - 1) / ALIGNMENT))

__global__ void tensor_product_forward_kernel(
    float *__restrict__ in1_global, float *__restrict__ in2_global, float *__restrict__ out,
    uint32_t *__restrict__ out_indices_global, float *__restrict__ cb_palette_global,
    uint64_t *__restrict__ input_indices_global, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t cb_palette_size, size_t out_size, size_t minibatch_size) {
  const size_t num_samples = min(minibatch_size - blockIdx.x * SAMPLES_PER_BLOCK, (size_t)SAMPLES_PER_BLOCK);
  if (threadIdx.x >= len_indices || blockIdx.x * SAMPLES_PER_BLOCK >= minibatch_size || num_samples <= 0) {
    return;
  }

  in1_global += blockIdx.x * SAMPLES_PER_BLOCK * in1_size;
  in2_global += blockIdx.x * SAMPLES_PER_BLOCK * in2_size;
  out += blockIdx.x * SAMPLES_PER_BLOCK * out_size;

  // Copy in1, in2, and the Clebsch-Gordon coefficient palette to shared memory.
  extern __shared__ uint8_t shared_mem[];
  float* cb_palette = (float*)shared_mem;
  uint32_t* input_indices = (uint32_t*)(cb_palette + cb_palette_size);
  uint32_t* out_indices = input_indices + MAX_GULP_SIZE * 9;
  float* in1 = (float*)(out_indices + MAX_GULP_SIZE);
  float* in2 = (float*)(out_indices + MAX_GULP_SIZE) + in1_size * SAMPLES_PER_BLOCK;
  for (int i = threadIdx.x; i < cb_palette_size; i += blockDim.x) {
    cb_palette[i] = cb_palette_global[i];
  }
  for (int i = threadIdx.x; i < num_samples * in1_size; i += blockDim.x) {
    in1[i] = in1_global[i];
  }
  for (int i = threadIdx.x; i < num_samples * in2_size; i += blockDim.x) {
    in2[i] = in2_global[i];
  }

  while (len_indices) {
    // Indices are the same for the entire block
    const size_t gulp_size = min(len_indices, (size_t)MAX_GULP_SIZE);
    for (int i = threadIdx.x; i < gulp_size; i += blockDim.x) {
      // Decompress our input and Clebsch-Gordon indices.
      uint64_t input_idx = input_indices_global[i];
      uint32_t in1_idx1 = input_idx & 0x3FF;
      uint32_t in2_idx1 = (input_idx >> 10) & 0x3FF;
      uint32_t cb_idx1 = (input_idx >> 20) & 0x3FF;
      uint32_t in1_idx2 = ((input_idx >> 32) & 0x1F) ^ in1_idx1;
      uint32_t in2_idx2 = ((input_idx >> 37) & 0x1) + in2_idx1;
      uint32_t cb_idx2 = (input_idx >> 38) & 0x3FF;
      uint32_t in1_idx3 = ((input_idx >> 48) & 0x1F) ^ in1_idx1;
      uint32_t in2_idx3 = ((input_idx >> 53) & 0x1) + in2_idx2;
      uint32_t cb_idx3 = input_idx >> 54;
      input_indices[9 * i] = in1_idx1;
      input_indices[9 * i + 1] = in2_idx1;
      input_indices[9 * i + 2] = cb_idx1;
      input_indices[9 * i + 3] = in1_idx2;
      input_indices[9 * i + 4] = in2_idx2;
      input_indices[9 * i + 5] = cb_idx2;
      input_indices[9 * i + 6] = in1_idx3;
      input_indices[9 * i + 7] = in2_idx3;
      input_indices[9 * i + 8] = cb_idx3;
      out_indices[i] = out_indices_global[i];
    }
    len_indices -= gulp_size;
    input_indices_global += gulp_size;
    out_indices_global += gulp_size;

    // Perform the actual tensor product.
    #pragma unroll
    for (int j = 0; j < num_samples; j++) {
      const float* curr_in1 = in1 + j * in1_size;
      const float* curr_in2 = in2 + j * in2_size;
      float* curr_out = out + j * out_size;
      #pragma unroll
      for (int i = threadIdx.x; i < gulp_size; i += blockDim.x) {
        // Perform the actual mul-adds.
        float acc = curr_in1[input_indices[9 * i]] * curr_in2[input_indices[9 * i + 1]] * cb_palette[input_indices[9 * i + 2]];
        acc += curr_in1[input_indices[9 * i + 3]] * curr_in2[input_indices[9 * i + 4]] * cb_palette[input_indices[9 * i + 5]];
        acc += curr_in1[input_indices[9 * i + 6]] * curr_in2[input_indices[9 * i + 7]] * cb_palette[input_indices[9 * i + 8]];

        // Write back out to our output.
        atomicAdd(curr_out + out_indices[i], acc);
      }
    }
  }
}

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    uint32_t *__restrict__ out_indices, float *__restrict__ cb_palette,
    uint64_t *__restrict__ input_indices, size_t len_indices, size_t in1_size,
    size_t in2_size, size_t cb_palette_size, size_t out_size, int batch_size) {
  while (batch_size > 0) {
      int minibatch_size = batch_size < MINIBATCH_MAX_SIZE ? batch_size : MINIBATCH_MAX_SIZE;
      tensor_product_forward_kernel<<<NUM_BLOCKS, NUM_THREADS, ((in1_size + in2_size) * SAMPLES_PER_BLOCK + cb_palette_size) * sizeof(float) + MAX_GULP_SIZE * 10 * sizeof(uint32_t)>>>(
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
