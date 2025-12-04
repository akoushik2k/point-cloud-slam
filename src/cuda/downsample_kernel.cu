/**
 * @file downsample_kernel.cu
 * @brief CUDA kernel for hashing points into a voxel grid structure.
 * This is the first step in GPU-accelerated Voxel Grid Downsampling.
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Simple point structure for the GPU
struct PointXYZI {
    float x, y, z;
    float intensity; // We include intensity for completeness
};

// Voxel hashing kernel: assigns each point to a unique voxel key
__global__ void voxel_hash_kernel(
    const PointXYZI* __restrict__ input_points, // Input point cloud
    unsigned int* __restrict__ output_hash,    // Output voxel hash key (one per point)
    int N,                                     // Number of points
    float voxel_size,                          // Voxel size (e.g., 0.2m)
    int max_key                                // Max possible hash key
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        PointXYZI p = input_points[idx];

        // 1. Quantize coordinates to voxel coordinates (i, j, k)
        int i = floorf(p.x / voxel_size);
        int j = floorf(p.y / voxel_size);
        int k = floorf(p.z / voxel_size);

        // 2. Simple linear hashing (mapping 3D index to 1D hash key)
        // This hash is for demonstration; production code uses a more robust spatial hash.
        // We assume indices i, j, k are well within the range [-max_key/2, max_key/2]
        output_hash[idx] = (unsigned int)(i * 1000000 + j * 1000 + k);
    }
}

// Host function (called from C++) to launch the kernel
// This function takes a raw pointer to point data (already on the CPU)
extern "C" void gpu_voxel_hash(
    const float* h_input_data, // Host pointer to (N, 4) point data
    unsigned int* h_output_hash, // Host pointer for output hash keys
    int N, 
    float voxel_size
) {
    // 1. Prepare Device Memory
    PointXYZI *d_input;
    unsigned int *d_hash;
    
    // N points * 4 floats (XYZI) * sizeof(float)
    size_t input_size = N * sizeof(PointXYZI);
    // N points * sizeof(unsigned int)
    size_t hash_size = N * sizeof(unsigned int);

    // Allocate Device memory for input points and output hash keys
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_hash, hash_size);

    // 2. Copy Host (CPU) memory to Device (GPU)
    // The input data is (N, 4) floats, which matches our PointXYZI structure
    cudaMemcpy(d_input, h_input_data, input_size, cudaMemcpyHostToDevice);

    // 3. Configure and Launch the Kernel
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // We use a large, arbitrary max_key for the simple hash demonstration
    int max_key = 100000000; 

    voxel_hash_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_hash, N, voxel_size, max_key);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    
    // 4. Copy Device (GPU) result back to Host (CPU)
    cudaMemcpy(h_output_hash, d_hash, hash_size, cudaMemcpyDeviceToHost);

    // 5. Cleanup
    cudaFree(d_input);
    cudaFree(d_hash);
}