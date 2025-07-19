import torch
from e3nn import o3
from coo_tp import COOTensorProduct
from einsum_tp import EinsumTensorProduct
from cuda_tp import CudaTensorProduct
import cuequivariance as cueq
import cuequivariance_torch as cueq_torch
import openequivariance as oeq
from openequivariance.benchmark.tpp_creation_utils import FullyConnectedTPProblem as FCTPP
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
        # nequip-revmd17-benzene triggers some sort of bug.
#        ('nequip-revmd17-benzene', '128x0e + 128x1e + 128x2e + 128x3e', '1x0e + 1x1e + 1x2e + 1x3e', '128x0e + 128x1e + 128x2e + 128x3e'),
        ('nequip-water', '64x0e + 64x1e', '1x0e + 1x1e', '64x0e + 64x1e'),
]

def full_tp_benchmark():
  print('==== Natural tensor product benchmarks ====')
  for config in configs:
    print(config[0])
    irreps_in1 = o3.Irreps(config[1])
    irreps_in2 = o3.Irreps(config[2])

    e3nn_tp = o3.FullTensorProduct(irreps_in1, irreps_in2)
    e3nn_tp.compile()
    test_tp = CudaTensorProduct(irreps_in1, irreps_in2)

    in1 = torch.randn((BATCH_SIZE, irreps_in1.dim))
    in2 = torch.randn((BATCH_SIZE, irreps_in2.dim))
    assert(float(torch.sum((e3nn_tp(in1, in2) - test_tp(in1, in2))**2)) / BATCH_SIZE / irreps_in1.dim / irreps_in2.dim < 1e-14)

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
      test_tp(in1, in2)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(0, NUM_TEST_ROUNDS):
      test_tp(in1, in2)
    torch.cuda.synchronize()
    end = time.time()
    throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
    print('test throughput: {:.2E}'.format(throughput))

def fc_tp_benchmark():
  print('==== Fully connected benchmarks ====')
  for config in configs:
    print(config[0])
    irreps_in1 = o3.Irreps(config[1])
    irreps_in2 = o3.Irreps(config[2])
    irreps_out = o3.Irreps(config[3])

    e3nn_tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    e3nn_tp.compile()
    test_tp = CudaTensorProduct(irreps_in1, irreps_in2, irreps_out)
    oeq_tp = FCTPP(irreps_in1, irreps_in2, irreps_out)
    oeq_tp = oeq.TensorProduct(oeq_tp, torch_op=True)
    cueq_tp = cueq_torch.FullyConnectedTensorProduct(cueq.Irreps(cueq.SO3, config[1].replace('e', '')),
                                               cueq.Irreps(cueq.SO3, config[2].replace('e', '')),
                                               cueq.Irreps(cueq.SO3, config[3].replace('e', '')), 
                                               device='cuda', internal_weights=False, shared_weights=True)

    in1 = torch.randn((BATCH_SIZE, irreps_in1.dim))
    in2 = torch.randn((BATCH_SIZE, irreps_in2.dim))
    weights = e3nn_tp.weight.reshape((1, test_tp.num_weights))
    assert(e3nn_tp(in1, in2).shape == test_tp(in1, in2, weights.flatten()).shape)
    assert(test_tp.num_weights == cueq_tp.weight_numel)
    # TODO: Add MSE asserts here once we get path normalization working.

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
      oeq_tp(in1, in2, weights)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(0, NUM_TEST_ROUNDS):
      oeq_tp(in1, in2, weights)
    torch.cuda.synchronize()
    end = time.time()
    throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
    print('oeq throughput: {:.2E}'.format(throughput))

    for i in range(0, NUM_WARMUP_ROUNDS):
      cueq_tp(in1, in2, weights)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(0, NUM_TEST_ROUNDS):
      cueq_tp(in1, in2, weights)
    torch.cuda.synchronize()
    end = time.time()
    throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
    print('cueq throughput: {:.2E}'.format(throughput))

    for i in range(0, NUM_WARMUP_ROUNDS):
      test_tp(in1, in2, weights.flatten())
    torch.cuda.synchronize()
    start = time.time()
    for i in range(0, NUM_TEST_ROUNDS):
      test_tp(in1, in2, weights.flatten())
    torch.cuda.synchronize()
    end = time.time()
    throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
    print('test throughput: {:.2E}'.format(throughput))

full_tp_benchmark()
print()
fc_tp_benchmark()
