// Very minimal skeleton for the kernel

#include <stdio.h>

# define INPUT_DIM 100
# define FILTER_DIM 5
# define CONV_OUT_DIM 20
# define OUT_NEURON_DIM 4000
# define CONV_LAYER_SIZE 10
# define OUT_LAYER_SIZE 10
# define THREAD_NUM 32

// Output Layer
extern "C" __global__ void output_kernel(
    double input[OUT_NEURON_DIM],
    double output[OUT_LAYER_SIZE][THREAD_NUM],
    double weight[OUT_LAYER_SIZE][OUT_NEURON_DIM]
) {
    int THREAD_MULTIPLIER = 125;

    double result = 0;

    int current_thread = threadIdx.x * THREAD_MULTIPLIER;
    int current_weight = blockIdx.x;

    #pragma unroll
    for (int x = 0; x < THREAD_MULTIPLIER; x++) {
        result += input[x + current_thread] * weight[current_weight][x + current_thread];
    }

    output[current_weight][current_thread / THREAD_MULTIPLIER] = result;
}

// Convolution_layer + Relu layer
extern "C" __global__ void convolution_relu_kernel(
    double filter[CONV_LAYER_SIZE][FILTER_DIM][FILTER_DIM],
    double input[INPUT_DIM][INPUT_DIM],
    double output[CONV_LAYER_SIZE][CONV_OUT_DIM][CONV_OUT_DIM]
) {

    int current = blockIdx.x;
    int current_x = threadIdx.x * FILTER_DIM;
    int current_y = threadIdx.y * FILTER_DIM;

    double result = 0;

    // Convolution Layer
    #pragma unroll
    for (int x = 0; x < FILTER_DIM; x++) {
        #pragma unroll
        for (int y = 0; y < FILTER_DIM; y++) {
            result += filter[current][x][y] * input[current_x + x][current_y + y];
        }
    }

    // Relu Layer
    output[current][current_x / FILTER_DIM][current_y / FILTER_DIM] = max(0.0, result);
}