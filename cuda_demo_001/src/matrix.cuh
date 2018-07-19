#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int add(int a, int b);

bool initCUDA();

int sumOfSquares(int* data, int DATA_SIZE);
#endif