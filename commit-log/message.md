# CUDA Implementation of CNN

## Summary
In this lab, we were given the implementation of a CNN that uses a CPU's cores to handle the processing.
Our goal was to re-implement this desing, except using the GPU's cores to handle processing. Due to the vast number of cores
that the GPU has over the CPU, we can parralelize our program more efficicently. In order to do this implementation,
a CUDA kernel was defined which allows us to access the memory of the GPU and give it programmatic functions to complete.

## Tech Details
The main goal of this lab is to replicate the processes that exist in cpu.rs into a cuda kernel. The CNN in the
cpu.rs file had three layers which were the convulution layer, relu layer, and the output layer. Two global functions were defined in the kernel to
replicate these layers. Since the relu layer was a very simple layer that only ensured that our result non-negative, it was decided to combine the
convulation layer that the relu layer into one global function.

The hardest component to replicate in each global function were the dot products. Using #pragma-unroll to ensure that all additions for a single vector cam happen at the same time and breaking down the component of each vector using `threadIdx.x` and `threadIdx.y`, the dot product functionality was achieved.

In order to use our global functions, we had to let rust have access to our GPU through the initialization of `rustacuda` API and then loading our ptx model and creating a data stream. Since CUDA code and memory safety can no longer be verififed by Rust's type system, the global function calls were wrapped in an `unsafe{}`.


## Correctness
In order to test my code for correctness, I did 10 iterations of generating a new input csv file and then running the cpu and cuda version of the CNN. The outputs of these processes were compared to one another using `generate.py`. It was found that there were no discepancies between the two output files for each iteration and hence, it was concluded that the cuda implementation of the CNN was correct.

## Testing for Performance
Perfomance was determined by comparing the computation time of both implementations. For the CPU implementation, the CNN took an average of 30000 microseconds. On the other hand, the CUDA implementation was 14000 microsends. Due to the high discrepancy between the two numbers,
it was concluded that the CUDA implementation had a better performance when compared to the CPU implementation. This can be attributed to the
number of cores that exist on the GPU and the parallelization that can be achieved.
