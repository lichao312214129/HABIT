from numba import prange
from numba import njit
import numpy as np
import time

@njit(parallel=True)
def compute_parallel(arr):
    for i in prange(arr.shape[0]):
        arr[i] = arr[i] ** 2

@njit
def compute_serial(arr):
    for i in range(arr.shape[0]):
        arr[i] = arr[i] ** 2

arr = np.random.rand(10000000)

start_time = time.time()
compute_parallel(arr)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(arr)


# none parallel
start_time = time.time()
compute_serial(arr)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(arr)
