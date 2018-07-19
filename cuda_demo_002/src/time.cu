#include "time.cuh"
#include <iostream>

__global__ static void time(int* gpudata, int* result, int DATA_SIZE) {
    int sum = 0;
    for (int i=0; i<DATA_SIZE; i++) {
        sum += gpudata[i] * gpudata[i] * gpudata[i];
    }
    *result = sum;
}

void time(int* data, int DATA_SIZE) {
    int* gpudata;
    int* result;

    cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));

    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    // test time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    time<<<1, 1, 0>>> (gpudata, result, DATA_SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    int sum_result = 0;
    cudaMemcpy(&sum_result, result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum result: " << sum_result << std::endl;
    std::cout << "Sum time: " << elapsedTime << std::endl;
    cudaFree(result);
    cudaFree(gpudata);
}