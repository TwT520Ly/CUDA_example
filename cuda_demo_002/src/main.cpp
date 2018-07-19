#include "time.cuh"
#include <iostream>

#define DATA_SIZE 10000
using namespace std;
int data[DATA_SIZE+5];

void generateNumbers(int* number, int size) {
    for (int i=0; i<size; i++) {
        number[i] = rand() % 10;
    }
}

int main() {

    generateNumbers(data, DATA_SIZE);
    time(data, DATA_SIZE);
    return 0;
}