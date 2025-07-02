import torch
from e3nn import o3
import math
import random

class EinsumTensorProduct(torch.nn.Module):
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
          cb_matrix_layout[l3].append((l1, l2, idx_in1, idx_in2))
          cb_height += 2 * l3 + 1
        idx_in2 += 2 * l2 + 1
      idx_in1 += 2 * l1 + 1

    self.cb_matrix = torch.zeros((cb_height, irreps_in1.dim, irreps_in2.dim))
    l3s = list(cb_matrix_layout.keys())
    l3s.sort()
    row_offset = 0
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
              self.cb_matrix[m3 + row_offset, m1 + in1_offset, m2 + in2_offset] = cb_coeffs[m1, m2, m3] * math.sqrt(2 * l3 + 1)
        row_offset += 2 * l3 + 1

  def forward(self, in1, in2):
    return torch.einsum('bi,bj,kij->bk', in1, in2, self.cb_matrix)
