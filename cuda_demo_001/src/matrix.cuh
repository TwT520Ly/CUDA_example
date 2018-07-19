#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int add(int a, int b);

bool initCUDA();

int sumOfSquares_gpu(int* data, int DATA_SIZE);

int sumOfSquares_cpu(int* data, int DATA_SIZE);

#endif