#ifndef MATRIX_CUH
#define MATRIX_CUH

bool initCUDA();

void gpuMatrix(const float * a, const float * b, float * gpuR, int n);
#endif