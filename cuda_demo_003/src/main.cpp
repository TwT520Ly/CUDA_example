#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "matrix.cuh"

#define MATRIX_SIZE 2000

using namespace std;

float * a;
float * b;
float * cpuR;
float * gpuR;

void genMat(float* t, int n) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            // 首先将数据归一化到[0, 1],防止在进行矩阵乘法的过程中出现溢出的情况,同时进行精通补充
            a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
        }
    }
}

int main() {

    if(!initCUDA()) {
        return 0;
    }

    int n = MATRIX_SIZE;

    a = (float*)malloc(sizeof(float) * n * n);
    b = (float*)malloc(sizeof(float) * n * n);
    cpuR = (float*)malloc(sizeof(float) * n * n);
    gpuR = (float*)malloc(sizeof(float) * n * n);

    srand(0);

    genMat(a, n);
    genMat(b, n);
    // GPU
    gpuMatrix(a, b, gpuR, n);
    // CPU
    clock_t start = clock();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n ;j++) {
            double t = 0.0;
            for (int k = 0; k < n; k++) {
                t += a[i * n + k] * b[k * n + j];
            }
            cpuR[i * n + j] = (float)t;
        }
    }
    clock_t end = clock();
    std::cout << "CPU time: " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

    // Error
    float max_error = 0.0;
    float average_err = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if(cpuR[i * n + j] != 0) {
                float err = fabs((cpuR[i * n + j] - gpuR[i * n + j]) / cpuR[i * n + j]);
                if (max_error < err) {
                    max_error = err;
                }

                average_err += err;
            }
        }
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Average error: " << average_err / (n * n) << std::endl;
    std::cout << cpuR[235] << " "<< gpuR[235] << std::endl;
    return 0;
}