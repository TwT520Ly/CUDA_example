#ifndef TIME_CUH
#define TIME_CUH

#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

void time(int* data, int DATA_SIZE);

#endif