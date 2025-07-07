import torch
from e3nn import o3
import math
import random
from torch.utils.cpp_extension import load

# In order to avoid branching, we only compress together indices which all
# contribute to the same output row. This helper function helps us determine
# how many of the next three indices we can actually cluster together.
def cluster_size(in1_idx1, in2_idx1, out_idx1,
                 in1_idx2, in2_idx2, out_idx2,
                 in1_idx3, in2_idx3, out_idx3):
  ret = 1
  if (in1_idx2 ^ in1_idx1) < 32 and (in2_idx2 - in2_idx1) < 2 and out_idx2 == out_idx1:
    ret += 1
    if (in1_idx3 ^ in1_idx1) < 32 and (in2_idx3 - in2_idx2) < 2 and out_idx3 == out_idx2:
      ret += 1

  return ret


# Greedy algorithm for clustering input indices together that are highly correlated.
# This allows us to use delta compression on the input indices and thus reduces how
# much space they take up. This also allows us to reduce the number of times we write
# back to the output row, since we can accumulate up to 3 mul-add operations before
# we need to write back.
def cluster_indices(in1_indices, in2_indices, cb_indices, out_indices):
  ret = []
  i = 0
  while i + 2 < len(in1_indices):
    size = cluster_size(in1_indices[i], in2_indices[i], out_indices[i],
                        in1_indices[i+1], in2_indices[i+1], out_indices[i+1],
                        in1_indices[i+2], in2_indices[i+2], out_indices[i+2])
    ret.append([])
    for j in range(0, size):
      ret[-1].append((in1_indices[i+j], in2_indices[i+j], cb_indices[i+j], out_indices[i+j]))
    i += size

  for j in range(i, len(in1_indices)):
    ret.append([])
    ret[-1].append((in1_indices[j], in2_indices[j], cb_indices[j], out_indices[j]))

  return ret


# Pack our index cluster into 64 bits.
def compress_cluster(cluster):
  ret = 0
  ret |= cluster[0][0]
  ret |= cluster[0][1] << 10
  ret |= cluster[0][2] << 20
  if len(cluster) > 1:
    ret |= (cluster[1][0] ^ cluster[0][0]) << 32
    ret |= (cluster[1][1] - cluster[0][1]) << 37
    ret |= cluster[1][2] << 38
    if len(cluster) > 2:
      # XOR with the first element in the cluster to avoid a data depenency on
      # the second element.
      ret |= (cluster[2][0] ^ cluster[0][0]) << 48
      ret |= (cluster[2][1] - cluster[1][1]) << 53
      ret |= cluster[2][2] << 54

  return ret


# Cluster and compress our input and Clebsch-Gordon indices.
def compress_indices(in1_indices, in2_indices, cb_indices, out_indices):
  input_indices = []
  ret_out_indices = []
  clusters = cluster_indices(in1_indices, in2_indices, cb_indices, out_indices)
  for i in range(0, len(clusters)):
    input_indices.append(compress_cluster(clusters[i]))
    ret_out_indices.append(clusters[i][0][3])

  for i in range(len(clusters), 256 * int((len(clusters) + 255) / 256)):
    input_indices.append(0)
    ret_out_indices.append(ret_out_indices[-1])

  return input_indices, ret_out_indices


class CudaTensorProduct(torch.nn.Module):
  def __init__(self, irreps_in1, irreps_in2):
    assert(irreps_in1.dim <= 1024)
    assert(irreps_in2.dim <= 1024)
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
    self.cb_palette.sort()
    self.cb_palette.insert(0, 0)
    self.cb_indices = list(map(lambda x: self.cb_palette.index(x), cb_vals))
    self.cb_palette = torch.tensor(self.cb_palette)

    # Delta compress the input indices and pack them with the compressed
    # Clebsch-Gordon coefficients.
    self.input_indices, self.out_indices = compress_indices(in1_indices, in2_indices, self.cb_indices, out_indices)
    self.input_indices = torch.tensor(self.input_indices, dtype=torch.uint64)

    # TODO: These out_indices are highly correlated with each other. I tried a few delta
    # compression schemes, but none of them play nice with the grid-stride loop.
    self.out_indices = torch.tensor(self.out_indices, dtype=torch.uint32)


  def forward(self, in1, in2):
    out = torch.zeros((in1.shape[0], in1.shape[1] * in2.shape[1]))
    return self.cuda_tp.forward(
      in1,
      in2,
      out,
      self.out_indices,
      self.cb_palette,
      self.input_indices)
