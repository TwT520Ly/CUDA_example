#include <iostream>
#include "matrix.cuh"

#define MAX_SIZE 10005
using namespace std;

int data[MAX_SIZE];
const int DATA_SIZE = 100;

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
    sum_gpu = sumOfSquares(data, DATA_SIZE);

    std::cout << "GPU sum: " << sum_gpu << std::endl;

    // 使用CPU进行测试
    int sum_cpu = 0;
    for (int i=0; i<DATA_SIZE; i++) {
        sum_cpu += data[i] * data[i] * data[i];
    }
    std::cout << "CPU sum: "<< sum_cpu << std::endl;
    return 0;
}

