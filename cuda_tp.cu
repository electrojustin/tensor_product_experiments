#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define LOG_NUM_THREADS 7
#define NUM_THREADS (1 << 7)
#define NUM_BLOCKS 1024
#define MINIBATCH_MAX_SIZE NUM_BLOCKS

#define BITFIELD_EXTRACT(SRC, DST, START, LEN) asm("bfe.u32 %0, %1, " #START ", " #LEN ";" : "=r"(DST) : "r"(SRC))
#define BITFIELD_EXTRACT_SIGNED(SRC, DST, START, LEN) asm("bfe.s32 %0, %1, " #START ", " #LEN ";" : "=r"(DST) : "r"(SRC))

__global__ void tensor_product_forward_kernel(
    float *__restrict__ in1_global, float *__restrict__ in2_global,
    float *__restrict__ out, float *__restrict__ cb_palette_global,
    uint32_t *__restrict__ block_jobs, uint32_t *__restrict__ block_job_sizes,
    size_t in1_size, size_t in2_size, size_t cb_palette_size, size_t out_size,
    size_t minibatch_size) {
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

  if (batch_idx >= minibatch_size || threadIdx.x >= out_size) {
    return;
  }

  // One iteration of this loop accumulates all the products needed for one
  // output row. This ensures we keep the number of writes at the absolute
  // minimum.
  for (int out_idx = threadIdx.x; out_idx < out_size; out_idx += blockDim.x) {
    // Unpack the instruction written in absolute form.
    uint32_t block_job_size = block_job_sizes[out_idx >> LOG_NUM_THREADS];
    uint32_t input_idx = block_jobs[threadIdx.x];
    int in1_idx = input_idx & 0x3FF;
    int in2_idx;
    BITFIELD_EXTRACT(input_idx, in2_idx, 10, 10);
    int cb_idx = input_idx >> 20;
    float acc = in1[in1_idx] * in2[in2_idx] * cb_palette[cb_idx];

    // Decompress the delta compressed instructions.
    for (int block_job_idx = threadIdx.x + blockDim.x;
         block_job_idx < block_job_size; block_job_idx += blockDim.x) {
      input_idx = block_jobs[block_job_idx];

      int in1_delta;
      BITFIELD_EXTRACT_SIGNED(input_idx, in1_delta, 0, 5);
      in1_idx += in1_delta;
      int in2_delta;
      BITFIELD_EXTRACT(input_idx, in2_delta, 5, 1);
      in2_idx += in2_delta;
      BITFIELD_EXTRACT(input_idx, cb_idx, 6, 10);
      acc += in1[in1_idx] * in2[in2_idx] * cb_palette[cb_idx];
      BITFIELD_EXTRACT_SIGNED(input_idx, in1_delta, 16, 5);
      in1_idx += in1_delta;
      BITFIELD_EXTRACT(input_idx, in2_delta, 21, 1);
      in2_idx += in2_delta;
      BITFIELD_EXTRACT(input_idx, cb_idx, 22, 10);
      acc += in1[in1_idx] * in2[in2_idx] * cb_palette[cb_idx];
    }
    out[out_idx] = acc;
    block_jobs += block_job_size;
  }
}

void tensor_product_forward_cuda(
    float *__restrict__ in1, float *__restrict__ in2, float *__restrict__ out,
    float *__restrict__ cb_palette, uint32_t *__restrict__ block_jobs,
    uint32_t *__restrict__ block_job_sizes, size_t in1_size, size_t in2_size,
    size_t cb_palette_size, size_t out_size, int batch_size) {
  while (batch_size > 0) {
      int minibatch_size = batch_size < MINIBATCH_MAX_SIZE ? batch_size : MINIBATCH_MAX_SIZE;
      tensor_product_forward_kernel<<<NUM_BLOCKS, NUM_THREADS,
                                      (in1_size + in2_size + cb_palette_size) *
                                          sizeof(float)>>>(
          in1, in2, out, cb_palette, block_jobs, block_job_sizes, in1_size,
          in2_size, cb_palette_size, out_size, minibatch_size);
      batch_size -= MINIBATCH_MAX_SIZE;
      in1 += in1_size * MINIBATCH_MAX_SIZE;
      in2 += in2_size * MINIBATCH_MAX_SIZE;
      out += out_size * MINIBATCH_MAX_SIZE;
  }
}
