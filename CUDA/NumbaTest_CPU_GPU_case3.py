# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:06:05 2023

@author: huanc
"""
from numba import cuda
import numpy as np
import math
import time as time
import multiprocessing as mp

def cpu_divide(a, b, result, n):
    for idx in range(n):
        result[idx] = a[idx] / b[idx]

def cpu_divide_api(a, b, result, n):
    result = np.divide(a, b)
    
    
def cpu_divide_mp_task(a, b, result):
    result = np.divide(x, y)
        
def cpu_divide_mp(a, b, result, n):
    process_list = []
    threadIdx = 0
    while(threadIdx<n):
        for i in range(1024):
            process_list.append(mp.Process(target=cpu_divide_mp_task, args = (a[i+threadIdx],b[i+threadIdx],result[i+threadIdx],)))
            process_list[i].start()
            if((i+threadIdx)>=(n-1)):
                break            
        if((i+threadIdx)>=(n-1)):
            break
        threadIdx = threadIdx + 1024
    for i in range(1024):
        process_list[i].join()

@cuda.jit
def gpu_divide(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx] = a[idx] / b[idx]

def testCPUandGPU(VectorDimention):
    for i in range(len(VectorDimention)):   
        n = VectorDimention[i]
        print("Test dimention: ", n)     
        x = np.arange(n).astype(np.int32)
        y = 2 * x * x + 1
        
        # copy to GPU momery
        x_device = cuda.to_device(x)
        y_device = cuda.to_device(y)
        # result momery
        gpu_result = cuda.device_array(n)
        cpu_result = np.empty(n)
        
        threads_per_block = 1024
        blocks_per_grid = math.ceil(n / threads_per_block)
        
        start = time.perf_counter()
        gpu_divide[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, n)
        cuda.synchronize()
        print("gpu vector divide time " + str(time.perf_counter() - start))
        start = time.perf_counter()
        cpu_result = np.divide(x, y)
        print("cpu vector np_divide time " + str(time.perf_counter() - start))
        if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
            print("result correct!")
        start = time.perf_counter()
        cpu_result = cpu_divide(x, y, cpu_result, n)
        print("cpu vector single_divide time " + str(time.perf_counter() - start))
        if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
            print("result correct!")
        start = time.perf_counter()
        #cpu_result = cpu_divide_mp(x, y, cpu_result, n)
        #print("cpu vector single_divide time " + str(time.perf_counter() - start))
        
            
        # Release GPU momery space
        #for_cleaing = cuda.get_current_device()
        #for_cleaing.reset()
