#include <iostream>
#include "matrix.cuh"
#include <ctime>
#include <typeinfo>

#define MAX_SIZE 2000000
using namespace std;

int data[MAX_SIZE];
const int DATA_SIZE = 1048576;

void generateNumbers(int* number, int size) {
    for (int i=0; i<size; i++) {
        number[i] = i % 5;
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
    sum_gpu = sumOfSquares_gpu(data, DATA_SIZE);
    std::cout << "GPU sum: " << sum_gpu << std::endl;
    std::cout << "***************" << std::endl;

    // 使用CPU进行测试
    int sum_cpu = 0;
    sum_cpu = sumOfSquares_cpu(data, DATA_SIZE);
    std::cout << "CPU sum: "<< sum_cpu << std::endl;

    return 0;
}

