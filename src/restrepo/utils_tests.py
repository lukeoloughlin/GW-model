import warnings
from typing import Any

import numpy as np
import numba.cuda as cuda

from params import RestrepoParams
from utils import *


def test_square():
    @cuda.jit
    def square_gpu(arr):
        idx = cuda.grid(1)
        if idx < len(arr):
            arr[idx] = square(arr[idx])

    arr_cpu = np.random.rand(10_000).astype(np.float32)
    arr_gpu = cuda.to_device(arr_cpu)

    arr_cpu2 = arr_cpu * arr_cpu
    square_gpu.forall(len(arr_cpu))(arr_gpu)
    cuda.synchronize()
    assert np.allclose(arr_cpu2, arr_gpu.copy_to_host())


def test_cube():
    @cuda.jit
    def cube_gpu(arr):
        idx = cuda.grid(1)
        if idx < len(arr):
            arr[idx] = cube(arr[idx])

    arr_cpu = np.random.rand(10_000).astype(np.float32)
    arr_gpu = cuda.to_device(arr_cpu)

    arr_cpu3 = arr_cpu * arr_cpu * arr_cpu
    cube_gpu.forall(len(arr_cpu))(arr_gpu)
    cuda.synchronize()
    assert np.allclose(arr_cpu3, arr_gpu.copy_to_host())


def test_pow4():
    @cuda.jit
    def pow4_gpu(arr):
        idx = cuda.grid(1)
        if idx < len(arr):
            arr[idx] = pow4(arr[idx])

    arr_cpu = np.random.rand(10_000).astype(np.float32)
    arr_gpu = cuda.to_device(arr_cpu)

    arr_cpu4 = arr_cpu * arr_cpu * arr_cpu * arr_cpu
    pow4_gpu.forall(len(arr_cpu))(arr_gpu)
    cuda.synchronize()
    assert np.allclose(arr_cpu4, arr_gpu.copy_to_host())


def test_calculate_rho(params: RestrepoParams):
    @cuda.jit
    def calc_rho_gpu(rho, cjsr, K, rho_inf):
        idx = cuda.grid(1)
        if idx < len(cjsr):
            rho[idx] = calculate_rho(cjsr[idx], K, rho_inf)

    K = params.K
    rho_inf = params.rho_inf

    cjsr_cpu = (700 + 100 * np.random.rand(10_000)).astype(np.float32)

    cjsr_gpu = cuda.to_device(cjsr_cpu)
    rho_gpu = cuda.device_array_like(cjsr_cpu)

    log_cjsr_K = np.log(cjsr_cpu) - np.log(K)
    rho_cpu = rho_inf / (1.0 + np.exp(23.0 * log_cjsr_K))

    calc_rho_gpu.forall(len(rho_cpu))(rho_gpu, cjsr_gpu, K, rho_inf)
    cuda.synchronize()
    assert np.allclose(rho_cpu, rho_gpu.copy_to_host())
    return rho_cpu


def test_calculate_Mhat(rho: Any, params: RestrepoParams):
    @cuda.jit
    def calc_Mhat_gpu(Mhat, rho_, BCSQN):
        idx = cuda.grid(1)
        if idx < len(rho_):
            Mhat[idx] = calculate_Mhat(rho_[idx], BCSQN)

    rho_gpu = cuda.to_device(rho)
    Mhat_gpu = cuda.device_array_like(rho)

    Mhat_cpu = (np.sqrt(1.0 + 8.0 * rho * params.BCSQN) - 1.0) / (
        4.0 * rho * params.BCSQN
    )

    calc_Mhat_gpu.forall(len(rho))(Mhat_gpu, rho_gpu, params.BCSQN)
    cuda.synchronize()
    assert np.allclose(Mhat_cpu, Mhat_gpu.copy_to_host())


def test_bubble_sort_ryr():
    @cuda.jit
    def bubble_sort_(arr_srted, arr):
        x, y = cuda.grid(2)
        if x < arr.shape[0] and y < arr.shape[1]:
            bubble_sort_ryr(arr, arr_srted, x, y)

    arr_cpu = np.random.rand(100, 100, 4).astype(np.float32)
    arr_gpu = cuda.to_device(arr_cpu)
    arr_srted = cuda.device_array_like(arr_cpu)

    tpb = (16, 16)
    bpg = (100 // 16 + 1, 100 // 16 + 1)
    bubble_sort_[bpg, tpb](arr_srted, arr_gpu)
    cuda.synchronize()
    arr_cpu.sort(axis=-1)
    assert np.allclose(arr_cpu, arr_srted.copy_to_host())


if __name__ == "__main__":
    params = RestrepoParams()
    warnings.filterwarnings("ignore")
    test_square()
    print("square test passed!")
    test_cube()
    print("cube test passed!")
    test_pow4()
    print("pow4 test passed!")
    rho = test_calculate_rho(params)
    print("calculate_rho test passed!")
    test_calculate_Mhat(rho, params)
    print("calculate_Mhat test passed!")
    test_bubble_sort_ryr()
    print("bubble_sort_ryr test passed!")
