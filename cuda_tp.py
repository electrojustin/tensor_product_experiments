import torch
from e3nn import o3
import math
import random
from torch.utils.cpp_extension import load

class CudaTensorProduct(torch.nn.Module):
  def __init__(self, irreps_in1, irreps_in2, irreps_out=None):
    assert(irreps_in1.dim <= 4096)
    assert(irreps_in2.dim <= 4096)
    assert(irreps_in1.lmax <= 15)
    assert(irreps_in2.lmax <= 15)
    super().__init__()
    self.cuda_tp = load(name='cuda_tp', sources=['cuda_tp.cpp', 'cuda_tp.cu'])
    cb_matrix_layout = {}
    self.cb_height = 0
    idx_in1 = 0
    for l1 in irreps_in1.ls:
      idx_in2 = 0
      for l2 in irreps_in2.ls:
        for l3 in range(abs(l1-l2), l1+l2+1):
          if irreps_out is not None and l3 not in irreps_out.ls:
            continue
          if l3 not in cb_matrix_layout:
            cb_matrix_layout[l3] = []
          cb_matrix_layout[l3].append((l1, l2, idx_in1, idx_in2))
          self.cb_height += 2 * l3 + 1
        idx_in2 += 2 * l2 + 1
      idx_in1 += 2 * l1 + 1

    self.flops = 0
    if irreps_out is not None:
      self.out_dims = irreps_out.dim
      self.num_weights = 0
      self.out_layout = {}
      gather_offset = 0
      self.gather_indices = []
      for l in irreps_out.ls:
        if l not in self.out_layout:
          self.out_layout[l] = [0]
        self.out_layout[l][0] += 1
      for l in set(irreps_out.ls):
        self.out_layout[l].append(len(cb_matrix_layout[l]))
        self.num_weights += self.out_layout[l][0] * self.out_layout[l][1]
        # 1 fused-multiply-add per weight in the dense matmul.
        self.flops += (2 * l + 1) * self.out_layout[l][0] * self.out_layout[l][1]
    else:
      self.num_weights = None

    l3s = list(cb_matrix_layout.keys())
    l3s.sort()
    row_offset = 0
    in1_indices = []
    in2_indices = []
    out_indices = []
    cb_vals = []
    if irreps_out is None:
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
                self.flops += 2 # 1 multiply and 1 fused-multiply-add.
          row_offset += 2 * l3 + 1
    else:
      # If we need to do linear mixing at the end, it's more efficient to break
      # the mixing up into dense matmul operations. In order to do this, we need
      # to output m values in planes rather than the normal interleaved form.
      row_offset = 0
      for l3 in l3s:
        multiplicities = cb_matrix_layout[l3]
        multiplicities.sort(key=lambda x: x[0] * irreps_in2.lmax + x[1])
        for m3 in range(0, 2 * l3 + 1):
          multiplicity_idx = 0
          for multiplicity in multiplicities:
            l1, l2, in1_offset, in2_offset = multiplicity
            cb_coeffs = o3.wigner_3j(l1, l2, l3)
            for m2 in range(0, 2 * l2 + 1):
              for m1 in range(0, 2 * l1 + 1):
                if cb_coeffs[m1, m2, m3] == 0:
                  continue
                in1_indices.append(m1 + in1_offset)
                in2_indices.append(m2 + in2_offset)
                out_indices.append(row_offset + m3 * len(multiplicities) + multiplicity_idx)
                cb_vals.append(float(cb_coeffs[m1, m2, m3] * math.sqrt(2 * l3 + 1)))
                self.flops += 2 # 1 multiply and 1 fused-multiply-add
            multiplicity_idx += 1
        row_offset += len(multiplicities) * (2 * l3 + 1)

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


  def forward(self, in1, in2, weights=None):
    assert((weights is None and self.num_weights is None) or (weights is not None and self.num_weights is not None))
    batch_size = in1.shape[0]
    padded_batch_size = self.samples_per_block * int((batch_size + self.samples_per_block - 1) / self.samples_per_block)
    padded_in1 = torch.zeros((padded_batch_size, in1.shape[1]))
    padded_in2 = torch.zeros((padded_batch_size, in2.shape[1]))
    padded_in1[:batch_size, :] = in1
    padded_in2[:batch_size, :] = in2
    intermediate = torch.zeros((padded_in1.shape[0], self.cb_height))
    intermediate = self.cuda_tp.forward(
      padded_in1,
      padded_in2,
      intermediate,
      self.cb_palette,
      self.block_jobs,
      self.block_job_sizes)[:batch_size, :]
    if weights is None:
      return intermediate
    else:
      intermediate = torch.t(intermediate)
      out = torch.zeros((batch_size, self.out_dims))
      ls = list(self.out_layout.keys())
      ls.sort()
      intermediate_idx = 0
      out_idx = 0
      weights_idx = 0
      gather_offset = 0
      gather_indices = []
      for l in ls:
        intermediate_muls = self.out_layout[l][1]
        out_muls = self.out_layout[l][0]
        weights_len = intermediate_muls * out_muls
        curr_weights = weights[weights_idx:weights_idx + weights_len].reshape((out_muls, intermediate_muls))
        for m in range(0, 2 * l + 1):
          out[:, out_idx:out_idx+out_muls] = \
                  torch.t(torch.matmul(curr_weights, intermediate[intermediate_idx:intermediate_idx+intermediate_muls, :]))
          intermediate_idx += intermediate_muls
          out_idx += out_muls
        # Create gather indices so we can re-interleave the m planes.
        for i in range(0, out_muls):
          for m in range(0, 2 * l + 1):
            gather_indices.append(gather_offset + m * out_muls + i)
        gather_offset += out_muls * (2 * l + 1)
        weights_idx += weights_len
      return out[:, gather_indices]
