#include "matrix.cuh"
#include <iostream>
#include <cstdio>
#include <cstdlib>


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

            if (prop.major >= 1) {
                break;
            }
        }
    }

    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x. \n");
        return false;
    }

}
