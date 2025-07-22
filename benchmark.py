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
ERROR_TOLERANCE = 1e-10

configs = [
        ('tetris-poly-1', '1x0e + 1x1o + 1x2e + 1x3o', '1x0e + 1x1o + 1x2e + 1x3o', '64x0e + 24x1e + 24x1o + 16x2e + 16x2o'),
        ('tetris-poly-2', '64x0e + 24x1e + 24x1o + 16x2e + 16x2o', '1x0e + 1x1o + 1x2e', '1x0o + 6x0e'),
        ('DiffDock-L=1', '10x1o + 10x1e + 48x0e + 48x0o', '1x0e + 1x1o', '10x1o + 10x1e + 48x0e + 48x0o'),
        ('DiffDock-L=2', '10x1o + 10x1e + 48x0e + 48x0o', '1x0e + 1x1o + 1x2e', '10x1o + 10x1e + 48x0e + 48x0o'),
        ('mace-large', '128x0e + 128x1o + 128x2e', '1x0e + 1x1o + 1x2e + 1x3o', '128x0e + 128x1o + 128x2e + 128x3o'),
        ('mace-medium', '128x0e + 128x1o', '1x0e + 1x1o + 1x2e + 1x3o', '128x0e + 128x1o + 128x2e'),
        ('nequip-lips', '32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e', '1x0e + 1x1o + 1x2e', '32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e'),
        ('nequip-revmd17-aspirin', '32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e', '1x0e + 1x1o', '64x0o + 64x0e + 64x1o + 64x1e'),
        ('nequip-revmd17-toluene', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e', '1x0e + 1x1o + 1x2e', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e'),
        # nequip-revmd17-benzene triggers some sort of bug.
#        ('nequip-revmd17-benzene', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e', '1x0e + 1x1o + 1x2e + 1x3o', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e'),
        ('nequip-water', '32x0o + 32x0e + 32x1o + 32x1e', '1x0e + 1x1o', '32x0o + 32x0e + 32x1o + 32x1e'),
]

def time_test(callback, name):
  for i in range(0, NUM_WARMUP_ROUNDS):
    callback()
  torch.cuda.synchronize()
  start = time.time()
  for i in range(0, NUM_TEST_ROUNDS):
    callback()
  torch.cuda.synchronize()
  end = time.time()
  throughput = BATCH_SIZE * NUM_TEST_ROUNDS / (end - start)
  print('{} throughput: {:.2E}'.format(name, throughput))


def full_tp_benchmark():
  print('==== Natural tensor product benchmarks ====')
  for config in configs:
    print(config[0])
    irreps_in1 = o3.Irreps(config[1]).sort().irreps.simplify()
    irreps_in2 = o3.Irreps(config[2]).sort().irreps.simplify()

    e3nn_tp = o3.FullTensorProduct(irreps_in1, irreps_in2)
    e3nn_tp.compile()
    test_tp = CudaTensorProduct(irreps_in1, irreps_in2)

    in1 = torch.randn((BATCH_SIZE, irreps_in1.dim))
    in2 = torch.randn((BATCH_SIZE, irreps_in2.dim))
    assert(float(torch.sum((e3nn_tp(in1, in2) - test_tp(in1, in2))**2)) / BATCH_SIZE / irreps_in1.dim / irreps_in2.dim < ERROR_TOLERANCE)

    def e3nn_cb():
      e3nn_tp(in1, in2)
    time_test(e3nn_cb, 'e3nn')

    def test_cb():
      test_tp(in1, in2)
    time_test(test_cb, 'test')


def fc_tp_benchmark():
  print('==== Fully connected benchmarks ====')
  for config in configs:
    print(config[0])
    irreps_in1 = o3.Irreps(config[1]).sort().irreps.simplify()
    irreps_in2 = o3.Irreps(config[2]).sort().irreps.simplify()
    irreps_out = o3.Irreps(config[3]).sort().irreps.simplify()

    start = time.time()
    e3nn_tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    e3nn_tp.compile()
    print('e3nn compile time: {:.2E}s'.format(time.time() - start))
    start = time.time()
    test_tp = CudaTensorProduct(irreps_in1, irreps_in2, irreps_out)
    print('test compile time: {:.2E}s'.format(time.time() - start))
    start = time.time()
#    cueq_tp = cueq_torch.FullyConnectedTensorProduct(cueq.Irreps(cueq.SO3, config[1].replace('e', '')),
#                                               cueq.Irreps(cueq.SO3, config[2].replace('e', '')),
#                                               cueq.Irreps(cueq.SO3, config[3].replace('e', '')), 
#                                               device='cuda', internal_weights=False, shared_weights=True)
    print('cueq compile time: {:.2E}s'.format(time.time() - start))
    start = time.time()
#    oeq_tp = FCTPP(irreps_in1, irreps_in2, irreps_out)
#    oeq_tp = oeq.TensorProduct(oeq_tp, torch_op=True)
    print('oeq compile time: {:.2E}s'.format(time.time() - start))
#    cueq_ir_mul_tp = cueq_torch.FullyConnectedTensorProduct(cueq.Irreps(cueq.O3, config[1].replace('e', '')),
#                                               cueq.Irreps(cueq.O3, config[2].replace('e', '')),
#                                               cueq.Irreps(cueq.O3, config[3].replace('e', '')),
#                                               layout=cueq.ir_mul, device='cuda',
#                                               internal_weights=False, shared_weights=True)

    in1 = torch.randn((BATCH_SIZE, irreps_in1.dim))
    in2 = torch.randn((BATCH_SIZE, irreps_in2.dim))
    weights = torch.tensor(e3nn_tp.weight).reshape((1, test_tp.num_weights))
    weights_perm = test_tp.convert_weights(weights.flatten())

    assert(e3nn_tp(in1, in2).shape == test_tp(in1, in2, weights.flatten()).shape)
    assert(float(torch.sum((e3nn_tp(in1, in2) - test_tp(in1, in2, weights_perm))**2)) / BATCH_SIZE / irreps_out.dim < ERROR_TOLERANCE)

    def e3nn_cb():
      e3nn_tp(in1, in2)
    time_test(e3nn_cb, 'e3nn')

#    def oeq_cb():
#      oeq_tp(in1, in2, weights)
#    time_test(oeq_cb, 'oeq')

#    def cueq_cb():
#      cueq_tp(in1, in2, weights)
#    time_test(cueq_cb, 'cueq')

#    def cueq_ir_mul_cb():
#      cueq_ir_mul_tp(in1, in2, weights)
#    time_test(cueq_ir_mul_cb, 'cueq (ir_mul layout)')

    def test_prepermute_cb():
      test_tp(in1, in2, weights_perm)
    time_test(test_prepermute_cb, 'test')


full_tp_benchmark()
print()
fc_tp_benchmark()
