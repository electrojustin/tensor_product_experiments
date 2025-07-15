import torch
from e3nn import o3
from coo_tp import COOTensorProduct
from einsum_tp import EinsumTensorProduct
from cuda_tp import CudaTensorProduct
import time

torch.set_default_device('cuda')

BATCH_SIZE = 50000
NUM_WARMUP_ROUNDS = 10
NUM_TEST_ROUNDS = 100

configs = [
        ('tetris-poly-1', '1x0e + 1x1e + 1x2e + 1x3e', '1x0e + 1x1e + 1x2e + 1x3e', '64x0e + 48x1e + 32x2e'),
        ('tetris-poly-2', '64x0e + 48x1e + 32x2e', '1x0e + 1x1e + 1x2e', '7x0e'),
        ('DiffDock-L=1', '96x0e + 20x1e', '1x0e + 1x1e', '96x0e + 20x1e'),
        ('DiffDock-L=2', '96x0e + 20x1e', '1x0e + 1x1e + 1x2e', '96x0e + 20x1e'),
        ('mace-large', '128x0e + 128x1e + 128x2e', '1x0e + 1x1e + 1x2e + 1x3e', '128x0e + 128x1e + 128x2e + 128x3e'),
        ('mace-medium', '128x0e + 128x1e', '1x0e + 1x1e + 1x2e + 1x3e', '128x0e + 128x1e + 128x2e'),
        ('nequip-lips', '64x0e + 64x1e + 64x2e', '1x0e + 1x1e + 1x2e', '64x0e + 64x1e + 64x2e'),
        ('nequip-revmd17-aspirin', '128x0e + 128x1e', '1x0e + 1x1e', '128x0e + 128x1e'),
        ('nequip-revmd17-toluene', '128x0e + 128x1e + 128x2e', '1x0e + 1x1e + 1x2e', '128x0e + 128x1e + 128x2e'),
        ('nequip-revmd17-benzene', '128x0e + 128x1e + 128x2e + 128x3e', '1x0e + 1x1e + 1x2e + 1x3e', '128x0e + 128x1e + 128x2e + 128x3e'),
        ('nequip-water', '64x0e + 64x1e', '1x0e + 1x1e', '64x0e + 64x1e'),
]

print('==== Natural tensor product benchmarks ====')
for config in configs:
  print(config[0])
  irreps_in1 = o3.Irreps(config[1])
  irreps_in2 = o3.Irreps(config[2])

  e3nn_tp = o3.FullTensorProduct(irreps_in1, irreps_in2)
  e3nn_tp.compile()
  cuda_tp = CudaTensorProduct(irreps_in1, irreps_in2)

  in1 = torch.randn((BATCH_SIZE, irreps_in1.dim))
  in2 = torch.randn((BATCH_SIZE, irreps_in2.dim))
  #print(str(float(torch.sum((e3nn_tp(in1, in2) - cuda_tp(in1, in2))**2)) / BATCH_SIZE / irreps_in1.dim / irreps_in2.dim))
  #assert(float(torch.sum((e3nn_tp(in1, in2) - cuda_tp(in1, in2))**2)) / BATCH_SIZE / irreps_in1.dim / irreps_in2.dim < 1e-14)

  for i in range(0, NUM_WARMUP_ROUNDS):
    e3nn_tp(in1, in2)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(0, NUM_TEST_ROUNDS):
    e3nn_tp(in1, in2)
  torch.cuda.synchronize()
  end = time.time()
  throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
  print('e3nn throughput: {:.2E}'.format(throughput))

  for i in range(0, NUM_WARMUP_ROUNDS):
    cuda_tp(in1, in2)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(0, NUM_TEST_ROUNDS):
    cuda_tp(in1, in2)
  torch.cuda.synchronize()
  end = time.time()
  throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
  print('cuda throughput: {:.2E}'.format(throughput))

print()

print('==== Fully connected benchmarks ====')
for config in configs:
  print(config[0])
  irreps_in1 = o3.Irreps(config[1])
  irreps_in2 = o3.Irreps(config[2])
  irreps_out = o3.Irreps(config[3])

  e3nn_tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
  e3nn_tp.compile()
  cuda_tp = CudaTensorProduct(irreps_in1, irreps_in2, irreps_out)

  in1 = torch.randn((BATCH_SIZE, irreps_in1.dim))
  in2 = torch.randn((BATCH_SIZE, irreps_in2.dim))
  weights = torch.randn((cuda_tp.num_weights))
  assert(e3nn_tp(in1, in2).shape == cuda_tp(in1, in2, weights).shape)

  for i in range(0, NUM_WARMUP_ROUNDS):
    e3nn_tp(in1, in2)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(0, NUM_TEST_ROUNDS):
    e3nn_tp(in1, in2)
  torch.cuda.synchronize()
  end = time.time()
  throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
  print('e3nn throughput: {:.2E}'.format(throughput))

  for i in range(0, NUM_WARMUP_ROUNDS):
    cuda_tp(in1, in2, weights)
  torch.cuda.synchronize()
  start = time.time()
  for i in range(0, NUM_TEST_ROUNDS):
    cuda_tp(in1, in2, weights)
  torch.cuda.synchronize()
  end = time.time()
  throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
  print('cuda throughput: {:.2E}'.format(throughput))
