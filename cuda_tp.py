import torch
from e3nn import o3
import math
import random
from torch.utils.cpp_extension import load

class CudaTensorProduct(torch.nn.Module):
  def __init__(self, irreps_in1, irreps_in2):
    assert(irreps_in1.dim <= 4096)
    assert(irreps_in2.dim <= 4096)
    assert(irreps_in1.lmax <= 15)
    assert(irreps_in2.lmax <= 15)
    super().__init__()
    self.cuda_tp = load(name='cuda_tp', sources=['cuda_tp.cpp', 'cuda_tp.cu'])
    cb_matrix_layout = {}
    cb_height = 0
    idx_in1 = 0
    for l1 in irreps_in1.ls:
      idx_in2 = 0
      for l2 in irreps_in2.ls:
        for l3 in range(abs(l1-l2), l1+l2+1):
          if l3 not in cb_matrix_layout:
            cb_matrix_layout[l3] = []
          cb_matrix_layout[l3].append((l1, l2, idx_in1, idx_in2))
          cb_height += 2 * l3 + 1
        idx_in2 += 2 * l2 + 1
      idx_in1 += 2 * l1 + 1

    l3s = list(cb_matrix_layout.keys())
    l3s.sort()
    row_offset = 0
    in1_indices = []
    in2_indices = []
    out_indices = []
    cb_vals = []
    for l3 in l3s:
      multiplicities = cb_matrix_layout[l3]
      multiplicities.sort(key=lambda x: x[0] * irreps_in2.lmax + x[1])
      for multiplicity in multiplicities:
        l1, l2, in1_offset, in2_offset = multiplicity
        cb_coeffs = o3.wigner_3j(l1, l2, l3)
        for m3 in range(0, 2 * l3 + 1):
          for m2 in range(0, 2 * l2 + 1):
            for m1 in range(0, 2 * l1 + 1):
              if cb_coeffs[m1, m2, m3] == 0:
                continue
              in1_indices.append(m1 + in1_offset)
              in2_indices.append(m2 + in2_offset)
              out_indices.append(m3 + row_offset)
              cb_vals.append(float(cb_coeffs[m1, m2, m3] * math.sqrt(2 * l3 + 1)))
        row_offset += 2 * l3 + 1

    # Palette compression for the Clebsch-Gordon coefficients.
    self.cb_palette = list(set(cb_vals))
    assert(len(self.cb_palette) <= 256)
    self.cb_palette.sort()
    self.cb_palette.insert(0, 0)
    self.cb_indices = list(map(lambda x: self.cb_palette.index(x), cb_vals))
    self.cb_palette = torch.tensor(self.cb_palette)

    # Batch up index tuples by their corresponding output row.
    metadata = {}
    for in1_idx, in2_idx, cb_idx, out_idx in zip(in1_indices, in2_indices, self.cb_indices, out_indices):
      if out_idx not in metadata:
        metadata[out_idx] = []
      elif in2_idx - metadata[out_idx][-1][1] > 1:
        # Usually delta in2_idx is either 0 or 1, but every once in a while
        # it's 2, so we add fake entries to the tuple list to bridge the gap.
        for i in range(metadata[out_idx][-1][1]+1, in2_idx):
          metadata[out_idx].append((in1_idx, i, 0, out_idx))
      metadata[out_idx].append((in1_idx, in2_idx, cb_idx, out_idx))

    # Create instruction list for every thread in a block, called a "block job".
    out_indices = list(set(out_indices))
    out_indices.sort()
    num_threads = 512
    block_jobs = []
    block_job_sizes = []
    while len(out_indices) > 0:
      block_job_size_in_clusters = min(len(out_indices), num_threads)
      # We need to pad the instructions because there might be a varying number
      # per output row.
      cluster_size = max(map(lambda x: len(metadata[x]), out_indices[:block_job_size_in_clusters]))
      block_job = []
      # We use cluster_size rows and num_threads columns even though this is a 
      # little unintuitive because it allows us to write the decompression loop
      # in grid-stride form, which helps with read coallescing.
      for i in range(0, cluster_size):
        for j in range(0, num_threads):
          if j >= block_job_size_in_clusters or i >= len(metadata[out_indices[j]]):
            # Insert padding for this instruction.
            if i == 0 or i % 2 == 1:
              block_job.append(0)
          else:
            metadata_entry = metadata[out_indices[j]][i]
            if i == 0:
              # The first instruction needs to use absolute addressing.
              block_job.append(metadata_entry[0] | (metadata_entry[1] << 12) | (metadata_entry[2] << 24))
            else:
              # Subsequent instructions can use a delta compression scheme
              # to pack two instructions into one 32-bit word.
              last_metadata_entry = metadata[out_indices[j]][i-1]
              block_job_entry = ((metadata_entry[0] - last_metadata_entry[0]) & 0x1F) \
                                  | ((metadata_entry[1] - last_metadata_entry[1]) << 5) \
                                  | (metadata_entry[2] << 6)
              if i % 2 == 1:
                block_job.append(block_job_entry)
              else:
                block_job[int(i / 2) * num_threads + j] |= block_job_entry << 16
      block_jobs += block_job
      block_job_sizes.append(len(block_job))
      out_indices = out_indices[block_job_size_in_clusters:]

    self.block_jobs = torch.tensor(block_jobs, dtype=torch.uint32)
    self.block_job_sizes = torch.tensor(block_job_sizes, dtype=torch.uint32)
    self.samples_per_block = 8


  def forward(self, in1, in2):
    batch_size = in1.shape[0]
    padded_batch_size = self.samples_per_block * int((batch_size + self.samples_per_block - 1) / self.samples_per_block)
    padded_in1 = torch.zeros((padded_batch_size, in1.shape[1]))
    padded_in2 = torch.zeros((padded_batch_size, in2.shape[1]))
    padded_in1[:batch_size, :] = in1
    padded_in2[:batch_size, :] = in2
    out = torch.zeros((padded_in1.shape[0], padded_in1.shape[1] * padded_in2.shape[1]))
    return self.cuda_tp.forward(
      padded_in1,
      padded_in2,
      out,
      self.cb_palette,
      self.block_jobs,
      self.block_job_sizes)[:batch_size, :]
