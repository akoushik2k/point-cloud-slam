/**
 * @file simple_kernel.cu
 * @brief Basic CUDA kernel for parallel vector addition (C = A + B).
 * * This confirms the NVCC compiler is working and targeting the GPU architecture.
 */
#include <stdio.h>

// Global function executed by the GPU (the kernel)
__global__ void vector_add(const float *A, const float *B, float *C, int N)
{
    // Calculate the global index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Host function (called from C++) to launch the kernel
extern "C" void run_cuda_test(float *h_A, float *h_B, float *h_C, int N) {
    // 1. Memory allocation on the Device (GPU)
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);
    
    // Allocate GPU memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 2. Copy Host (CPU) memory to Device (GPU)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3. Configure the kernel launch
    // We aim for 256 threads per block, and calculate the necessary grid size
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch the kernel
    vector_add<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize the GPU with the CPU (wait for kernel to finish)
    cudaDeviceSynchronize();

    // 5. Copy Device (GPU) result back to Host (CPU)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 6. Cleanup device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}