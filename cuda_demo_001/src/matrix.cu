#include "matrix.cuh"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#define MAX_THREAD 512
#define MAX_BLOCK 64

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


__global__ static void sumOfSquares(int* num, int* result, int DATA_SIZE) {
    // 获取线程编号
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    int sum = 0;
    for (int i=bid * MAX_THREAD + tid; i< DATA_SIZE; i += MAX_THREAD * MAX_BLOCK) {
        sum += num[i] * num[i] * num[i];
    }
    result[tid + bid * MAX_THREAD] = sum;
}

int sumOfSquares_gpu(int* data, int DATA_SIZE) {
    int* gpudata;
    int* result;

    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int) * MAX_THREAD * MAX_BLOCK);

    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // test time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    sumOfSquares<<<MAX_BLOCK, MAX_THREAD, 0>>>(gpudata, result, DATA_SIZE);

    int sum_gpu[MAX_THREAD * MAX_BLOCK];

    cudaMemcpy(&sum_gpu, result, sizeof(int) * MAX_THREAD * MAX_BLOCK, cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i=0; i<MAX_THREAD * MAX_BLOCK; i++) {
        sum += sum_gpu[i];
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    std::cout << "GPU time: " <<  elapsedTime << std::endl;
    cudaFree(gpudata);
    cudaFree(result);

    return sum;
}

int sumOfSquares_cpu(int* data, int DATA_SIZE) {
    int sum_cpu = 0;
    clock_t start_cpu = clock();
    for (int i=0; i<DATA_SIZE; i++) {
        sum_cpu += data[i] * data[i] * data[i];
    }

    clock_t end_cpu = clock();

    std::cout << "CPU time: " << (double)(end_cpu - start_cpu)* 1000.0 / (CLOCKS_PER_SEC) << std::endl;
    return sum_cpu;
}