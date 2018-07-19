#include "matrix.cuh"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int add(int a, int b) {
    int c;
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    add <<<1, 1>>> (a, b, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    return c;
}

bool initCUDA() {
    int count;

    // 获取cuda数目
    cudaGetDeviceCount(&count);

    std::cout << "Cuda number: "<< count << std::endl;

    if (count == 0) {
        fprintf(stderr, "There is no device\n");
        return false;
    }
    int i;

    for (i=0; i<count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            // std::cout << prop.multiProcessorCount << std::endl;
            std::cout << prop.clockRate << std::endl;
            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x. \n");
        return false;
    }

    cudaSetDevice(i);
    return true;
}


__global__ static void sumOfSquares(int* num, int* result, int DATA_SIZE, clock_t* time) {
    int sum = 0;
    clock_t start_gpu = clock();
    for (int i=0; i<DATA_SIZE; i++) {
        sum += num[i] * num[i] * num[i];
    }
    clock_t end_gpu = clock();
    *result = sum;
    *time = end_gpu - start_gpu;
}

int sumOfSquares_gpu(int* data, int DATA_SIZE) {
    int* gpudata;
    int* result;
    clock_t* time;

    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));
    cudaMalloc((void**)&time, sizeof(clock_t));

    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    sumOfSquares<<<1, 1, 0>>>(gpudata, result, DATA_SIZE, time);

    int sum_gpu;
    clock_t time_gpu;

    cudaMemcpy(&sum_gpu, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_gpu, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    // clockRate: 1582000 kHZ
    std::cout << "GPU time: " << (double)(time_gpu) / (1582000 * 1000.0) << std::endl;

    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    return sum_gpu;
}

int sumOfSquares_cpu(int* data, int DATA_SIZE) {
    int sum_cpu = 0;
    clock_t start_cpu = clock();
    for (int i=0; i<DATA_SIZE; i++) {
        sum_cpu += data[i] * data[i] * data[i];
    }

    clock_t end_cpu = clock();

    std::cout << "CPU time: " << (double)(end_cpu - start_cpu) / (CLOCKS_PER_SEC ) << std::endl;
    return sum_cpu;
}