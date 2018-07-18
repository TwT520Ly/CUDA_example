#include "matrix.cuh"
#include <iostream>
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
}