#include "matrix.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include "device_launch_parameters.h"

#define THREAD_NUM 256
#define MATRIX_SIZE 2000

const int blocks_num = MATRIX_SIZE * (MATRIX_SIZE + THREAD_NUM - 1);

bool initCUDA() {
    int count;
    cudaGetDeviceCount(&count);

    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;

    for(i = 0; i< count; i++) {
        cudaDeviceProp prop;

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

__global__ static void multiCUDA(const float * a, const float * b, float * c, int n) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // 暂时没有看懂
    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / n;
    const int column = idx % n;

    if (row < n && column < n) {
        float t = 0;
        for (int i = 0; i < n; i++) {
            t += a[row * n + i] * b[i * n + column];
        }

        c[row * n + column] = t;
    }
}

void gpuMatrix(const float * a, const float * b, float * gpuR, int n) {
    float * cuda_a;
    float * cuda_b;
    float * cuda_c;


    cudaMalloc((void**)&cuda_a, sizeof(float) * n * n);
    cudaMalloc((void**)&cuda_b, sizeof(float) * n * n);
    cudaMalloc((void**)&cuda_c, sizeof(float) * n * n);

    cudaMemcpy(cuda_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // test time
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    multiCUDA<<< blocks_num, THREAD_NUM, 0 >>>(cuda_a, cuda_b, cuda_c, n);

    cudaEventRecord(stop, 0);
    float elapsedTime;

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "GPU time: " << elapsedTime << std::endl;

    cudaMemcpy(gpuR, cuda_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

}


