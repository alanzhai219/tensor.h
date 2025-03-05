#include "tensor_cu.h"  // The new CUDA-enabled tensor.h
#include <time.h>
#ifdef _WIN32
    #include <windows.h>
    #include <winsock.h> // This includes timeval definition
#else
    #include <sys/time.h>
#endif
#include <cuda_runtime.h>

#define M_PI 3.14159265358979323846

void get_time(struct timeval *t) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    t->tv_sec = (long)(counter.QuadPart / frequency.QuadPart);
    t->tv_usec = (long)((counter.QuadPart % frequency.QuadPart) * 1000000 / frequency.QuadPart);
#else
    gettimeofday(t, NULL);
#endif
}

void load_csv(Tensor* x, Tensor* y, char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(1);
    }

    char line[10000];
    char *token;
    
    for(int b = 0; b < 60000; b++) {
        if(fgets(line, sizeof(line), file) != NULL) {
            token = strtok(line, ",");
            for(int i = 0; i < 28*28 + 10; i++) {
                if (token == NULL) {
                    fprintf(stderr, "CSV format error: not enough columns\n");
                    fclose(file);
                    exit(1);
                }
                float val = atof(token);
                if(i < 28*28) {
                    x->data->values[b * 28 * 28 + i] = val;
                } else {
                    y->data->values[b * 10 + (i - 28*28)] = val * (-1.0f);
                }
                token = strtok(NULL, ",");
            }
        } else {
            fprintf(stderr, "Not enough data for the specified batch size.\n");
            break;
        }
    }

    fclose(file);
}

void get_random_batch(Tensor* batch_x, Tensor* batch_y, Tensor* x, Tensor* y, int B) {
    static int seeded = 0;
    if (!seeded) {
        srand(0);
        seeded = 1;
    }
    
    int *used_indices = (int *)calloc(x->data->shape[0], sizeof(int));
    
    for(int i = 0; i < B; i++) {
        int index;
        do {
            index = rand() % x->data->shape[0];
        } while(used_indices[index]);
        used_indices[index] = 1;

        for(int j = 0; j < 784; j++) {
            int x_index = index * x->data->strides[0] + j;
            int batch_x_index = i * batch_x->data->strides[0] + j;
            batch_x->data->values[batch_x_index] = x->data->values[x_index];
        }

        for(int k = 0; k < 10; k++) {
            int y_index = index * y->data->strides[0] + k * y->data->strides[1];
            int batch_y_index = i * batch_y->data->strides[0] + k * batch_y->data->strides[1];
            batch_y->data->values[batch_y_index] = y->data->values[y_index];
        }
    }

    free(used_indices);
    // Transfer batch data to GPU
    cpu_to_cuda(batch_x->data);
    cpu_to_cuda(batch_y->data);
    cpu_to_cuda(batch_x->grad);
    cpu_to_cuda(batch_y->grad);
}

// CUDA kernel for weight updates
__global__ void update_weights_kernel(float* w, float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&w[i], -lr * grad[i]);
        grad[i] = 0.0f;
    }
}

void update_weights(Tensor* w, float lr) {
    int numBlocks = (w->data->size + BLOCKSIZE - 1) / BLOCKSIZE;
    update_weights_kernel<<<numBlocks, BLOCKSIZE>>>(w->data->cuda_values, w->grad->cuda_values, lr, w->data->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in weight update: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

float random_normal() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
}

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

float rand_range(float min, float max) {
    return min + rand_float() * (max - min);
}

float kaiming_uniform(int fan_in) {
    float gain = sqrtf(2.0f);  // for ReLU activation
    float std = gain / sqrtf(fan_in);
    float bound = sqrtf(3.0f) * std;
    return rand_range(-bound, bound);
}

// kaiming initialization
float kaiming_init(int fan_in) {
    float std_dev = sqrtf(2.0f / fan_in);
    return random_normal() * std_dev;
}

int main() {
    cudaSetDevice(0);
    
    int x_shape[] = {60000, 784};
    int y_shape[] = {60000, 10};
    Tensor* x = create_zero_tensor(x_shape, 2);
    Tensor* y = create_zero_tensor(y_shape, 2);

    load_csv(x, y, "mnist_train.csv");
    printf("loaded csv\n");
    
    // Original print after CUDA transfer
    cuda_to_cpu(x->data);
    cuda_to_cpu(y->data);

    int w1_shape[] = {784, 128};
    int w2_shape[] = {128, 10};
    Tensor* w1 = create_zero_tensor(w1_shape, 2);
    Tensor* w2 = create_zero_tensor(w2_shape, 2);

    for (int i = 0; i < w1->data->size; i++) w1->data->values[i] = kaiming_uniform(784);
    for (int i = 0; i < w2->data->size; i++) w2->data->values[i] = kaiming_uniform(128);
    cpu_to_cuda(w1->data);
    cpu_to_cuda(w2->data);
    cpu_to_cuda(w1->grad);
    cpu_to_cuda(w2->grad);

    int B = 128;
    float lr = 0.005;
    int batch_x_shape[] = {B, 784};
    int batch_y_shape[] = {B, 10};
    Tensor* batch_x = create_zero_tensor(batch_x_shape, 2);
    Tensor* batch_y = create_zero_tensor(batch_y_shape, 2);

    get_random_batch(batch_x, batch_y, x, y, B);
    cuda_to_cpu(batch_x->data);
    cuda_to_cpu(batch_y->data);
    
    struct timeval start, end;
    double elapsed_time;
    get_time(&start);
    printf("Start Time: %ld.%06ld seconds\n", start.tv_sec, start.tv_usec);

    for (int i = 0; i < 5000; i++) {
        get_random_batch(batch_x, batch_y, x, y, B);
        Tensor* w1_out = matmul(batch_x, w1);
        Tensor* relu_out = relu(w1_out);
        Tensor* w2_out = matmul(relu_out, w2);
        Tensor* lout = logsoftmax(w2_out);
        Tensor* mul_out = mul(lout, batch_y);
        Tensor* loss = mean(mul_out);
        // loss grad = 1
        float one = 1.0f;
        cudaMemcpy(loss->grad->cuda_values, &one, sizeof(float), cudaMemcpyHostToDevice);
        if (i == 0) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error after loss grad set: %s\n", cudaGetErrorString(err));
                exit(1);
            }
        }
            
        cudaMemset(w1->grad->cuda_values, 0, w1->grad->size * sizeof(float));
        cudaMemset(w2->grad->cuda_values, 0, w2->grad->size * sizeof(float));
        backward(loss);

        if (i % 100 == 0) {
            cuda_to_cpu(loss->data);  // Get loss value back to CPU for printing
            printf("batch: %d loss: %f \n", i, loss->data->values[0]);
        }

        update_weights(w1, lr);
        update_weights(w2, lr);

        free_tensor(w1_out);
        free_tensor(relu_out);
        free_tensor(w2_out);
        free_tensor(lout);
        free_tensor(mul_out);
        free_tensor(loss);
        
        cudaDeviceSynchronize();  // Ensure all operations complete
    }

    get_time(&end);
    printf("End Time:   %ld.%06ld seconds\n", end.tv_sec, end.tv_usec);

    elapsed_time = (end.tv_sec - start.tv_sec) + 
                   (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    // Clean up CUDA resources
    free_tensor(x);
    free_tensor(y);
    free_tensor(w1);
    free_tensor(w2);
    free_tensor(batch_x);
    free_tensor(batch_y);

    cudaDeviceReset();
    return 0;
}