#ifndef AUXILIARY_H
#define AUXILIARY_H
// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
// Includes STD libs
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cstdarg>
#include <random>
#include <unordered_map>
#include <cassert>
#include <math.h>
#include <cstdio>
#include <sstream>
#include <tuple>
#include <stdio.h>
#include <functional>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define NEIGHBOR_NUM 8
// DEBUG
#define DEBUG true
#define DEBUG_POINT_IDX 0

#define CHECK_CUDA \
{\
    auto ret = cudaDeviceSynchronize(); \
    if (ret != cudaSuccess) { \
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "(line " << __LINE__ << ") in Function: " << __FUNCTION__ << "\n" << cudaGetErrorString(ret); \
        throw std::runtime_error(cudaGetErrorString(ret)); \
    }\
}
#define M_PI 3.14159265358979323846

#endif //AUXILIARY_H
