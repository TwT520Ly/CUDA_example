cmake_minimum_required(VERSION 3.5)
project(Test2)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)
file(GLOB_RECURSE TEST_HEAD ${CMAKE_CURRENT_LIST_DIR}/src/.h ${CMAKE_CURRENT_LIST_DIR}/src/.cuh)
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp ${CMAKE_CURRENT_LIST_DIR}/src/*.cu)
set(CUDA_NVCC_FLAGS "-g -G")

cuda_add_executable(Test2 ${TEST_SRC} ${TEST_HEAD})

