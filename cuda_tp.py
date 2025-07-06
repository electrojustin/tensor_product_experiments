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

    test = {}
    for i in range(0, len(in1_indices)):
      if out_indices[i] not in test:
        test[out_indices[i]] = 0
      test[out_indices[i]] += 1
    print(sum(test.values())/len(test.keys()))

    self.cb_lut = list(set(cb_vals))
    self.cb_lut.sort()
    self.cb_lut.insert(0, 0)
    self.cb_indices = list(map(lambda x: self.cb_lut.index(x), cb_vals))
    #for i in range(len(in1_indices), 1024 * int((len(in1_indices) + 1023) / 1024)):
    #  in1_indices.append(0)
    #  in2_indices.append(0)
    #  out_indices.append(0)
    #  self.cb_indices.append(0)
    self.in1_indices = torch.tensor(in1_indices, dtype=torch.int64)
    self.in2_indices = torch.tensor(in2_indices, dtype=torch.int64)
    self.out_indices = torch.tensor(out_indices, dtype=torch.int64)
    self.out_indices_compressed = torch.zeros(self.out_indices.shape, dtype=torch.int64) 
    for i in range(0, self.in1_indices.shape[0]-1):
      self.out_indices_compressed[i] = int(self.out_indices[i+1]) - int(self.out_indices[i])
    self.cb_lut = torch.tensor(self.cb_lut)
    self.input_indices = ((torch.tensor(self.cb_indices, dtype=torch.int64) << 20)
                     | (self.out_indices_compressed << 31)
                     | (self.in1_indices << 10)
                     | self.in2_indices).to(dtype=torch.uint32)
    self.out_indices = self.out_indices.to(dtype=torch.uint32)


  def forward(self, in1, in2):
    out = torch.zeros((in1.shape[0], in1.shape[1] * in2.shape[1]))
    return self.cuda_tp.forward(
      in1,
      in2,
      out,
      self.out_indices,
      self.cb_lut,
      self.input_indices)
