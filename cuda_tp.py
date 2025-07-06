import torch
from e3nn import o3
import math
import random
from torch.utils.cpp_extension import load

class CudaTensorProduct(torch.nn.Module):
  def __init__(self, irreps_in1, irreps_in2):
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

    # Palette compress the non-zero Clebsch-Gordon coefficients.
    self.cb_palette = list(set(cb_vals))
    self.cb_palette.sort()
    self.cb_palette.insert(0, 0)
    self.cb_indices = list(map(lambda x: self.cb_palette.index(x), cb_vals))

    # For some reason PyTorch doesn't support shifting with uint32 cuda tensors.
    self.in1_indices = torch.tensor(in1_indices, dtype=torch.int64)
    self.in2_indices = torch.tensor(in2_indices, dtype=torch.int64)
    self.out_indices = torch.tensor(out_indices, dtype=torch.int64)
    self.cb_palette = torch.tensor(self.cb_palette)

    # Pack our input indices and compressed Clebsch-Gordon coefficients into one uint32.
    self.input_indices = ((torch.tensor(self.cb_indices, dtype=torch.int64) << 20)
                     | (self.in1_indices << 10)
                     | self.in2_indices).to(dtype=torch.uint32)

    # TODO: These out_indices are highly correlated with each other. I tried a few delta
    # compression schemes, but none of them play nice with the grid-stride loop. The same
    # is true to a lesser extent with the input indices.
    self.out_indices = self.out_indices.to(dtype=torch.uint32)


  def forward(self, in1, in2):
    out = torch.zeros((in1.shape[0], in1.shape[1] * in2.shape[1]))
    return self.cuda_tp.forward(
      in1,
      in2,
      out,
      self.out_indices,
      self.cb_palette,
      self.input_indices)
