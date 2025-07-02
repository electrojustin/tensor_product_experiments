import torch
from e3nn import o3
import math
import random

class COOTensorProduct(torch.nn.Module):
  def __init__(self, irreps_in1, irreps_in2):
    super().__init__()
    cb_matrix_layout = {}
    cb_height = 0
    idx_in1 = 0
    for l1 in irreps_in1.ls:
      idx_in2 = 0
      for l2 in irreps_in2.ls:
        for l3 in range(abs(l1-l2), l1+l2+1):
          if l3 not in cb_matrix_layout:
            cb_matrix_layout[l3] = []
          cb_matrix_layout[l3].append((l1, l2, idx_in1 * irreps_in2.dim + idx_in2))
          cb_height += 2 * l3 + 1
        idx_in2 += 2 * l2 + 1
      idx_in1 += 2 * l1 + 1

    cb_matrix_coords = [[], []]
    cb_matrix_vals = []

    l3s = list(cb_matrix_layout.keys())
    l3s.sort()
    row_offset = 0
    for l3 in l3s:
      multiplicities = cb_matrix_layout[l3]
      multiplicities.sort(key=lambda x: x[0] * irreps_in2.lmax + x[1])
      for multiplicity in multiplicities:
        l1, l2, col_offset = multiplicity
        cb_coeffs = o3.wigner_3j(l1, l2, l3)
        for m3 in range(0, 2 * l3 + 1):
          for m2 in range(0, 2 * l2 + 1):
            for m1 in range(0, 2 * l1 + 1):
              if cb_coeffs[m1, m2, m3] == 0:
                continue
              cb_matrix_coords[0].append(m3 + row_offset)
              cb_matrix_coords[1].append(m1 * irreps_in2.dim + m2 + col_offset)
              cb_matrix_vals.append(cb_coeffs[m1, m2, m3] * math.sqrt(2 * l3 + 1))
        row_offset += 2 * l3 + 1

    self.cb_matrix = torch.sparse_coo_tensor(cb_matrix_coords, cb_matrix_vals, (cb_height, irreps_in1.dim * irreps_in2.dim))

  def forward(self, in1, in2):
    outer = torch.t(torch.einsum('bi,bj->bij', in1, in2).reshape((in1.shape[0], -1)))
    return torch.t(torch.matmul(self.cb_matrix, outer))

