#include <iostream>
#include "matrix.cuh"
using namespace std;

int main() {
    // test the function -- add
    cout << add(1, 2) << endl;
    // init the cuda
    if (!initCUDA()) {
       return 0;
    }

    printf("CUDA initialized.\n");

    return 0;
}

