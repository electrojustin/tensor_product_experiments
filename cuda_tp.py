import torch
from e3nn import o3
import math
import random
from torch.utils.cpp_extension import load

# We need to serialize our outputs in a very particular order or we won't be
# compatible with e3nn
def sort_multiplicities(muls):
  grouped_muls = []
  for mul in muls:
    if len(grouped_muls) == 0 or grouped_muls[-1][0][4] != mul[4] or grouped_muls[-1][0][5] != mul[5]:
      grouped_muls.append([])
    grouped_muls[-1].append(mul)
  ret = []
  for group in grouped_muls:
    group.sort(key=lambda x: (x[0] + 1) * 16 + x[1])
    ret += group
  return ret

class CudaTensorProduct(torch.nn.Module):
  def __init__(self, irreps_in1, irreps_in2, irreps_out=None):
    assert(irreps_in1.dim <= 4096)
    assert(irreps_in2.dim <= 4096)
    assert(irreps_in1.lmax <= 15)
    assert(irreps_in2.lmax <= 15)
    super().__init__()

    self.cuda_tp = load(name='cuda_tp', sources=['cuda_tp.cpp', 'cuda_tp.cu'])

    if irreps_out is not None:
      weight_e3nn_indices_dict = {}
      self.e3nn_idx = 0
      self.out_dims = irreps_out.dim
      self.num_weights = 0
      self.out_matrix_layout = {}
      for mul, (l, p) in irreps_out:
        self.out_matrix_layout[(l, int((p + 1) / 2))] = [mul, 0]

    cb_matrix_layout = {}
    self.cb_height = 0
    irrep_offset1 = 0
    for mul1, (l1, p1) in irreps_in1:
      irrep_offset2 = 0
      for mul2, (l2, p2) in irreps_in2:
        for l3 in range(abs(l1-l2), l1+l2+1):
          p3 = int((p1 * p2 + 1) / 2)
          offset1 = irrep_offset1
          for i in range(0, mul1):
            offset2 = irrep_offset2
            for j in range(0, mul2):
              if irreps_out is not None and (l3, p3) not in self.out_matrix_layout:
                continue
              if (l3, p3) not in cb_matrix_layout:
                cb_matrix_layout[(l3, p3)] = []
              cb_matrix_layout[(l3, p3)].append((l1, l2, offset1, offset2, int((p1 + 1) / 2), int((p2 + 1) / 2)))
              self.cb_height += 2 * l3 + 1

              if irreps_out is not None:
                self.out_matrix_layout[(l3, p3)][1] += 1
                if (l3, p3) not in weight_e3nn_indices_dict:
                  weight_e3nn_indices_dict[(l3, p3)] = []
                weight_e3nn_indices_dict[(l3, p3)].append([])
                for k in range(0, self.out_matrix_layout[(l3, p3)][0]):
                  weight_e3nn_indices_dict[(l3, p3)][-1].append(self.e3nn_idx)
                  self.e3nn_idx += 1
                  self.num_weights += 1

              offset2 += 2 * l2 + 1
            offset1 += 2 * l1 + 1
        irrep_offset2 += (2 * l2 + 1) * mul2
      irrep_offset1 += (2 * l1 + 1) * mul1

    if irreps_out is not None:
      self.weight_e3nn_indices = []
      ls = list(weight_e3nn_indices_dict.keys())
      ls.sort(key=lambda x: 2 * (x[0] + 1) + x[1])
      for (l, p) in ls:
        # The weights are serialized in some weird column major order...
        self.weight_e3nn_indices += torch.t(torch.tensor(weight_e3nn_indices_dict[(l, p)])).flatten().tolist()
    else:
      self.num_weights = None

    l3s = list(cb_matrix_layout.keys())
    row_offset = 0
    in1_indices = []
    in2_indices = []
    out_indices = []
    cb_vals = []
    l3s.sort(key=lambda x: 2 * (x[0] + 1) + x[1])
    if irreps_out is None:
      for (l3, p3) in l3s:
        multiplicities = cb_matrix_layout[(l3, p3)]
        #multiplicities = sort_multiplicities(multiplicities)
        for multiplicity in multiplicities:
          l1, l2, in1_offset, in2_offset, _, _ = multiplicity
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
    else:
      # If we need to do linear mixing at the end, it's more efficient to break
      # the mixing up into dense matmul operations. In order to do this, we need
      # to output m values in planes rather than the normal interleaved form.
      row_offset = 0
      for (l3, p3) in l3s:
        multiplicities = cb_matrix_layout[(l3, p3)]
        for m3 in range(0, 2 * l3 + 1):
          multiplicity_idx = 0
          for multiplicity in multiplicities:
            l1, l2, in1_offset, in2_offset, _, _ = multiplicity
            cb_coeffs = o3.wigner_3j(l1, l2, l3)
            for m2 in range(0, 2 * l2 + 1):
              for m1 in range(0, 2 * l1 + 1):
                if cb_coeffs[m1, m2, m3] == 0:
                  continue
                in1_indices.append(m1 + in1_offset)
                in2_indices.append(m2 + in2_offset)
                out_indices.append(row_offset + m3 * len(multiplicities) + multiplicity_idx)
                cb_vals.append(float(cb_coeffs[m1, m2, m3] * math.sqrt(2 * l3 + 1)))
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


  def convert_weights(self, weights):
    return weights[self.weight_e3nn_indices]


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
      ls = list(self.out_matrix_layout.keys())
      ls.sort(key=lambda x: 2 * (x[0] + 1) + x[1])
      intermediate_idx = 0
      out_idx = 0
      weights_idx = 0
      gather_offset = 0
      gather_indices = []
      for (l, p) in ls:
        intermediate_muls = self.out_matrix_layout[(l, p)][1]
        out_muls = self.out_matrix_layout[(l, p)][0]
        weights_len = intermediate_muls * out_muls
        curr_weights = weights[weights_idx:weights_idx + weights_len].reshape((out_muls, intermediate_muls))
        curr_weights *= math.sqrt(1 / intermediate_muls)
        for m in range(0, 2 * l + 1):
          out[:, out_idx:out_idx+out_muls] = \
                  torch.t(torch.matmul(curr_weights, intermediate[intermediate_idx:intermediate_idx+intermediate_muls, :]))
          intermediate_idx += intermediate_muls
          out_idx += out_muls
        # Create gather indices so we can re-interleave the m planes.
        # TODO: Generate these in preprocessing.
        for i in range(0, out_muls):
          for m in range(0, 2 * l + 1):
            gather_indices.append(gather_offset + m * out_muls + i)
        gather_offset += out_muls * (2 * l + 1)
        weights_idx += weights_len
      return out[:, gather_indices]
