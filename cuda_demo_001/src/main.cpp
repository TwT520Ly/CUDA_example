#include <iostream>
#include "matrix.cuh"
#include <ctime>
#include <typeinfo>

#define MAX_SIZE 1000005
using namespace std;

int data[MAX_SIZE];
const int DATA_SIZE = 1000000;

void generateNumbers(int* number, int size) {
    for (int i=0; i<size; i++) {
        number[i] = rand() % 10;
    }
}
int main() {
    // test the function -- add
    cout << add(1, 2) << endl;
    // init the cuda
    if (!initCUDA()) {
       return 0;
    }

    printf("CUDA initialized.\n");

    generateNumbers(data, DATA_SIZE);

    // 使用GPU进行测试
    int sum_gpu;
    clock_t start_gpu = clock();
    sum_gpu = sumOfSquares(data, DATA_SIZE);
    clock_t end_gpu = clock();
    std::cout << "GPU sum: " << sum_gpu << std::endl;
    std::cout << (end_gpu - start_gpu) << std::endl;

    // 使用CPU进行测试
    int sum_cpu = 0;
    clock_t start_cpu = clock();
    for (int i=0; i<DATA_SIZE; i++) {
        sum_cpu += data[i] * data[i] * data[i];
    }
    clock_t end_cpu = clock();
    std::cout << "CPU sum: "<< sum_cpu << std::endl;
    std::cout << (end_cpu - start_cpu) << std::endl;

    return 0;
}

