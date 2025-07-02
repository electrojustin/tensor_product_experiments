import torch
from e3nn import o3
import math
import random

class WeightedTensorProduct(torch.nn.Module):
  def __init__(self, irreps_in1, irreps_in2, irreps_out=None, sparse=False):
    super().__init__()
    cb_matrix_layout = {}
    cb_height = 0
    idx_in1 = 0
    max_l3 = (irreps_out.lmax if irreps_out is not None else irreps_in1.lmax + irreps_in2.lmax) + 1
    for l1 in irreps_in1.ls:
      idx_in2 = 0
      for l2 in irreps_in2.ls:
        for l3 in range(abs(l1-l2), min(l1+l2+1, max_l3)):
          if l3 not in cb_matrix_layout:
            cb_matrix_layout[l3] = []
          cb_matrix_layout[l3].append((l1, l2, idx_in1 * irreps_in2.dim + idx_in2))
          cb_height += 2 * l3 + 1
        idx_in2 += 2 * l2 + 1
      idx_in1 += 2 * l1 + 1

    if not sparse:
      self.cb_matrix = torch.zeros((cb_height, irreps_in1.dim * irreps_in2.dim))
    else:
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
              if not sparse:
                self.cb_matrix[m3 + row_offset, m1 * irreps_in2.dim + m2 + col_offset] = cb_coeffs[m1, m2, m3] * math.sqrt(2 * l3 + 1)
              else:
                cb_matrix_coords[0].append(m3 + row_offset)
                cb_matrix_coords[1].append(m1 * irreps_in2.dim + m2 + col_offset)
                cb_matrix_vals.append(cb_coeffs[m1, m2, m3] * math.sqrt(2 * l3 + 1))
        row_offset += 2 * l3 + 1

    if sparse:
      self.cb_matrix = torch.sparse_coo_tensor(cb_matrix_coords, cb_matrix_vals, (cb_height, irreps_in1.dim * irreps_in2.dim))

    if irreps_out is not None:
      out_multiplicities = {}
      for l_out in irreps_out.ls:
        if l_out not in out_multiplicities:
          out_multiplicities[l_out] = 0
        out_multiplicities[l_out] += 1

      num_weights = 0
      for l3 in l3s:
        num_weights += out_multiplicities[l3] * len(cb_matrix_layout[l3])

      out_l3s = list(out_multiplicities.keys())
      out_l3s.sort()
      row_idx = 0
      weight_idx = 0
      weight_matrix_mask_coords = [[], []]
      weight_matrix_mask_vals = []
      for out_l3 in out_l3s:
        for out_multiplicity in range(0, out_multiplicities[out_l3]):
          col_idx = 0
          for l3 in l3s:
            for cb_multiplicity in range(0, len(cb_matrix_layout[l3])):
              if out_l3 == l3:
                for m in range(0, 2 * l3 + 1):
                    weight_matrix_mask_coords[0].append((row_idx + m) * cb_height + col_idx + m)
                    weight_matrix_mask_coords[1].append(weight_idx)
                    weight_matrix_mask_vals.append(1.0)
                weight_idx += 1
              col_idx += 2 * l3 + 1
          row_idx += 2 * out_l3 + 1

      self.weight_matrix_mask = torch.sparse_coo_tensor(weight_matrix_mask_coords, weight_matrix_mask_vals, (irreps_out.dim * cb_height, num_weights))
      weight_matrix_mask_coords[0], weight_matrix_mask_coords[1] = weight_matrix_mask_coords[1], weight_matrix_mask_coords[0]
      self.weight_matrix_mask_T = torch.sparse_coo_tensor(weight_matrix_mask_coords, weight_matrix_mask_vals, (irreps_out.dim * cb_height, num_weights))
      self.weight_matrix_shape = (irreps_out.dim, cb_height)
      self.weight_matrix = torch.nn.Parameter(torch.matmul(self.weight_matrix_mask, torch.randn((num_weights))).reshape(self.weight_matrix_shape).to_dense())
      self.register_full_backward_hook(self.backward_hook)
    else:
      self.weight_matrix = None

  def backward_hook(self, module, grad_input, grad_output):
    self.weight_matrix.grad = torch.matmul(self.weight_matrix_mask, torch.matmul(self.weight_matrix_mask_T, self.weight_matrix.grad.flatten())).reshape(self.weight_matrix_shape)

  def forward(self, in1, in2):
    if len(in1.shape) == 1:
      outer = torch.outer(in1, in2).flatten()
    else:
      outer = torch.t(torch.einsum('bi,bj->bij', in1, in2).reshape((in1.shape[0], -1)))
    if self.weight_matrix is None:
      return torch.matmul(self.cb_matrix, outer)
    else:
      return torch.t(torch.matmul(self.weight_matrix, torch.matmul(self.cb_matrix, outer)))
