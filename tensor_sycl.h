#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sycl/sych.hpp>

typedef struct {
    float* values;      // CPU data
    float* sycl_values; // GPU data
    int* shape;
    int* strides;
    int ndim;
    int size;
} Arr;

typedef union {
    int ival;
    float fval;
    int* ilist;
} Arg;

typedef struct Tensor {
    Arr* data;
    Arr* grad;
    int op;
    struct Tensor* prevs[MAX_PREVS];
    int num_prevs;
    Arg args[MAX_ARGS];
} Tensor;

// SYCL kernel declarations
void matmul_kernel(float* A, float* B, float* C, int M, int N, int K, sycl::nd_item<2>& ndi);

// Add these after the kernel declarations in the tensor_sycl.h
void matmul_kernel(float* A, float* B, float* C, int M, int N, int K, sycl::nd_item<2>& ndi) {
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// utils
void cpu_to_sycl(Arr* a, sycl::queue q) {
    if (a->sycl_values) return;
    a->sycl_values = sycl::make_device<float>(a->size, q);
    if (a->sycl_values) {
        printf("SYCL malloc error\n");
        // printf("SYCL malloc error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    q.memcpy(a->sycl_values, a->values, a->size * sizeof(float));
    // err = cudaMemcpy(a->cuda_values, a->values, a->size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to device error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
