import torch
from e3nn import o3
from coo_tp import COOTensorProduct
from einsum_tp import EinsumTensorProduct
from cuda_tp import CudaTensorProduct
import time

torch.set_default_device('cuda')

BATCH_SIZE = 20000
NUM_WARMUP_ROUNDS = 10
NUM_TEST_ROUNDS = 100

irreps_in1 = o3.Irreps('8x0e + 8x1e + 8x2e')
irreps_in2 = o3.Irreps('8x0e + 8x1e + 8x2e')

def test_tp(baseline, test, name):
  in1 = torch.randn((BATCH_SIZE, irreps_in1.dim))
  in2 = torch.randn((BATCH_SIZE, irreps_in2.dim))
  print(name + ' rmse: ' + str(torch.sum((test(in1, in2) - baseline(in1, in2))**2) / BATCH_SIZE / irreps_in1.dim / irreps_in2.dim))

  for i in range(0, NUM_WARMUP_ROUNDS):
    baseline(in1, in2)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(0, NUM_TEST_ROUNDS):
    baseline(in1, in2)
  torch.cuda.synchronize()
  end = time.time()
  print('baseline: ' + str(end - start))

  for i in range(0, NUM_WARMUP_ROUNDS):
    test(in1, in2)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(0, NUM_TEST_ROUNDS):
    test(in1, in2)
  torch.cuda.synchronize()
  end = time.time()
  print(name + ': ' + str(end - start))


baseline = o3.FullTensorProduct(irreps_in1, irreps_in2)
baseline.compile()
einsum_tp = EinsumTensorProduct(irreps_in1, irreps_in2)
coo_tp = COOTensorProduct(irreps_in1, irreps_in2)
cuda_tp = CudaTensorProduct(irreps_in1, irreps_in2)

#test_tp(baseline, einsum_tp, 'Einsum')
test_tp(baseline, coo_tp, 'Sparse Matmul')
test_tp(baseline, cuda_tp, 'Cuda')
