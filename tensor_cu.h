#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128
#define MAX_PREVS 3
#define MAX_ARGS 5
#define MAX_PARAM_TENSORS 10
// op codes
#define MATMUL 0
#define MEAN 1
#define MUL 2
#define RELU 3
#define LOGSOFTMAX 4

typedef struct {
    float* values;      // CPU data
    float* cuda_values; // GPU data
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

// CUDA kernel declarations
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K);
__global__ void matmul_transpose(float* A, float* B, float* C, int M, int N, int K);
__global__ void transpose_matmul(float* A, float* B, float* C, int M, int N, int K);
__global__ void relu_kernel(float* out, float* inp, size_t n);
__global__ void relu_backward_kernel(float* dinp, float* dout, float* out, size_t n);
__global__ void logsoftmax_kernel(float* out, float* inp, int B, int C, int strides_0, int strides_1);
__global__ void logsoftmax_backward_kernel(float* dinp, float* dout, float* out, int B, int C);
__global__ void mul_kernel(float* out, float* a, float* b, size_t n);
__global__ void mean_kernel(float* out, float* inp, size_t n);
__global__ void mean_backward_kernel(float* dinp, float* dout, size_t n);

// Add these after the kernel declarations in tensor_cu.h

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transpose(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[col * K + k];  // B is transposed
        }
        C[row * N + col] = sum;
    }
}

__global__ void transpose_matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k * M + row] * B[k * N + col];  // A is transposed
        }
        C[row * N + col] = sum;
    }
}

__global__ void relu_kernel(float* out, float* inp, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaxf(inp[i], 0.0f);
    }
}

__global__ void relu_backward_kernel(float* dinp, float* dout, float* out, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dinp[i] = (out[i] > 0.0f) ? dout[i] : 0.0f;
    }
}

__global__ void logsoftmax_kernel(float* out, float* inp, int B, int C, int strides_0, int strides_1) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_val = -INFINITY;
        // find max value
        for (int c = 0; c < C; c++) {
            float val = inp[b * strides_0 + c * strides_1];
            max_val = fmaxf(max_val, val);
        }
        
        // compute sum of exp
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += expf(inp[b * strides_0 + c * strides_1] - max_val);
        }
        
        // compute log softmax
        float logsum = logf(sum);
        for (int c = 0; c < C; c++) {
            out[b * strides_0 + c * strides_1] = 
                inp[b * strides_0 + c * strides_1] - max_val - logsum;
        }
    }
}

__global__ void logsoftmax_backward_kernel(float* dinp, float* dout, float* out, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float sum_dout = 0.0f;
        // first compute sum of dout
        for (int c = 0; c < C; c++) {
            sum_dout += dout[b * C + c];
        }
        // then compute gradient
        for (int c = 0; c < C; c++) {
            float sm = expf(out[b * C + c]);
            dinp[b * C + c] = dout[b * C + c] - sm * sum_dout;
        }
    }
}

// utils
cudaError_t checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    cudaDeviceSynchronize();
    return error;
}

void cpu_to_cuda(Arr* a) {
    if (a->cuda_values) return;
    cudaError_t err = cudaMalloc((void**)&a->cuda_values, a->size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpy(a->cuda_values, a->values, a->size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy to device error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void cuda_to_cpu(Arr* a) {
    if (!a->cuda_values) return;
    cudaMemcpy(a->values, a->cuda_values, a->size * sizeof(float), cudaMemcpyDeviceToHost);
}

Arr* create_arr(float* data, int* shape, int ndim) {
    Arr* arr = (Arr*)malloc(sizeof(Arr));
    arr->ndim = ndim;
    arr->shape = (int*)malloc(ndim * sizeof(int));
    arr->strides = (int*)malloc(ndim * sizeof(int));
    memcpy(arr->shape, shape, ndim * sizeof(int));
    
    arr->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = arr->size;
        arr->size *= shape[i];
    }
    
    arr->values = (float*)calloc(arr->size, sizeof(float));
    memcpy(arr->values, data, arr->size * sizeof(float));
    arr->cuda_values = NULL;
    cpu_to_cuda(arr);
    return arr;
}

Arr* create_arr_zeros(int* shape, int ndim) {
    Arr* arr = (Arr*)malloc(sizeof(Arr));
    arr->ndim = ndim;
    arr->shape = (int*)malloc(ndim * sizeof(int));
    arr->strides = (int*)malloc(ndim * sizeof(int));
    memcpy(arr->shape, shape, ndim * sizeof(int));
    
    arr->size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        arr->strides[i] = arr->size;
        arr->size *= shape[i];
    }
    
    arr->values = (float*)calloc(arr->size, sizeof(float));
    arr->cuda_values = NULL;
    // cpu_to_cuda(arr);
    return arr;
}

Tensor* create_tensor(float* data, int* shape, int ndim) {
    Arr* d = create_arr(data, shape, ndim);
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

Tensor* create_zero_tensor(int* shape, int ndim) {
    Arr* d = create_arr_zeros(shape, ndim);
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = d;
    t->grad = create_arr_zeros(shape, ndim);
    t->op = -1;
    t->num_prevs = 0;
    return t;
}

void free_arr(Arr* a) {
    if (a) {
        free(a->values);
        if (a->cuda_values) cudaFree(a->cuda_values);
        free(a->shape);
        free(a->strides);
        free(a);
    }
}

void free_tensor(Tensor* t) {
    if (t) {
        free_arr(t->data);
        free_arr(t->grad);
        free(t);
    }
}

// CUDA kernel implementations
__global__ void mul_kernel(float* out, float* a, float* b, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void mean_kernel(float* out, float* inp, size_t n) {
    __shared__ float sharedSum;
    if (threadIdx.x == 0) sharedSum = 0.0f;
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += inp[i];
    }
    atomicAdd(&sharedSum, sum);
    __syncthreads();
    
    if (threadIdx.x == 0) out[0] = sharedSum / n;
}

__global__ void mean_backward_kernel(float* dinp, float* dout, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dinp[i] = dout[0] / n;
}

// operations
Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* t = create_zero_tensor(a->data->shape, a->data->ndim);
    cpu_to_cuda(t->data);
    cpu_to_cuda(t->grad);
    int numBlocks = (t->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    mul_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values, a->data->cuda_values, b->data->cuda_values, t->data->size);
    checkCudaError(cudaGetLastError());
    
    t->op = MUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

Tensor* mean(Tensor* t) {
    int shape[] = {1};
    Tensor* m = create_zero_tensor(shape, 1);
    cpu_to_cuda(m->data);
    cpu_to_cuda(m->grad);
    mean_kernel<<<1, BLOCKSIZE>>>(m->data->cuda_values, t->data->cuda_values, t->data->size);
    checkCudaError(cudaGetLastError());
    
    m->op = MEAN;
    m->num_prevs = 1;
    m->prevs[0] = t;
    return m;
}

Tensor* matmul(Tensor* a, Tensor* b) {
    int M = a->data->shape[0];
    int K = a->data->shape[1];
    int N = b->data->shape[1];
    int shape[] = {M, N};
    Tensor* t = create_zero_tensor(shape, 2);
    cpu_to_cuda(t->data);
    cpu_to_cuda(t->grad);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(a->data->cuda_values, b->data->cuda_values, t->data->cuda_values, M, N, K);
    checkCudaError(cudaGetLastError());
    
    t->op = MATMUL;
    t->num_prevs = 2;
    t->prevs[0] = a;
    t->prevs[1] = b;
    return t;
}

Tensor* relu(Tensor* inp) {
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    cpu_to_cuda(t->data);
    cpu_to_cuda(t->grad);
    int numBlocks = (t->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    relu_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values, inp->data->cuda_values, t->data->size);
    checkCudaError(cudaGetLastError());
    
    t->op = RELU;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

Tensor* logsoftmax(Tensor* inp) {
    Tensor* t = create_zero_tensor(inp->data->shape, inp->data->ndim);
    cpu_to_cuda(t->data);
    cpu_to_cuda(t->grad);
    int numBlocks = (inp->data->shape[0] + BLOCKSIZE - 1) / BLOCKSIZE;
    logsoftmax_kernel<<<numBlocks, BLOCKSIZE>>>(t->data->cuda_values, inp->data->cuda_values, 
        inp->data->shape[0], inp->data->shape[1], inp->data->strides[0], inp->data->strides[1]);
    checkCudaError(cudaGetLastError());
    
    t->op = LOGSOFTMAX;
    t->num_prevs = 1;
    t->prevs[0] = inp;
    return t;
}

// backward implementations
void mul_backward(Tensor* out) {
    int numBlocks = (out->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    mul_kernel<<<numBlocks, BLOCKSIZE>>>(out->prevs[0]->grad->cuda_values, out->grad->cuda_values, 
        out->prevs[1]->data->cuda_values, out->data->size);
    mul_kernel<<<numBlocks, BLOCKSIZE>>>(out->prevs[1]->grad->cuda_values, out->grad->cuda_values, 
        out->prevs[0]->data->cuda_values, out->data->size);
    checkCudaError(cudaGetLastError());
}

void mean_backward(Tensor* out) {
    int numBlocks = (out->prevs[0]->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    mean_backward_kernel<<<numBlocks, BLOCKSIZE>>>(out->prevs[0]->grad->cuda_values, 
        out->grad->cuda_values, out->prevs[0]->data->size);
    checkCudaError(cudaGetLastError());
}

void matmul_backward(Tensor* out) {
    int P = out->prevs[0]->data->shape[0];
    int Q = out->prevs[0]->data->shape[1];
    int R = out->prevs[1]->data->shape[1];
    
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocksA((Q + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (P + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_transpose<<<numBlocksA, threadsPerBlock>>>(out->grad->cuda_values, 
        out->prevs[1]->data->cuda_values, out->prevs[0]->grad->cuda_values, P, Q, R);
    
    dim3 numBlocksB((R + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Q + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_matmul<<<numBlocksB, threadsPerBlock>>>(out->prevs[0]->data->cuda_values, 
        out->grad->cuda_values, out->prevs[1]->grad->cuda_values, Q, R, P);
    checkCudaError(cudaGetLastError());
}

void relu_backward(Tensor* out) {
    int numBlocks = (out->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    relu_backward_kernel<<<numBlocks, BLOCKSIZE>>>(out->prevs[0]->grad->cuda_values, 
        out->grad->cuda_values, out->data->cuda_values, out->data->size);
    checkCudaError(cudaGetLastError());
}

void logsoftmax_backward(Tensor* out) {
    int numBlocks = (out->data->shape[0] + BLOCKSIZE - 1) / BLOCKSIZE;
    logsoftmax_backward_kernel<<<numBlocks, BLOCKSIZE>>>(out->prevs[0]->grad->cuda_values, 
        out->grad->cuda_values, out->data->cuda_values, out->data->shape[0], out->data->shape[1]);
    checkCudaError(cudaGetLastError());
}

void backward(Tensor* t) {
    if (t->op == MUL) mul_backward(t);
    else if (t->op == MEAN) mean_backward(t);
    else if (t->op == MATMUL) matmul_backward(t);
    else if (t->op == RELU) relu_backward(t);
    else if (t->op == LOGSOFTMAX) logsoftmax_backward(t);
    
    for (int i = 0; i < t->num_prevs; i++) {
        backward(t->prevs[i]);
    }
}

void print_tensor(Tensor* t) {
    cuda_to_cpu(t->data);
    cuda_to_cpu(t->grad);
    printf("Tensor(\n");
    printf("\tdata: ");
    for (int i = 0; i < t->data->size; i++) printf("%f,", t->data->values[i]);
    printf("\n\tshape: ");
    for (int i = 0; i < t->data->ndim; i++) printf("%d,", t->data->shape[i]);
    printf("\n\tgrad: ");
    for (int i = 0; i < t->data->size; i++) printf("%f,", t->grad->values[i]);
    printf("\n)\n");
}