/**
 * @file gpu_processing.cpp
 * @brief C++ wrapper using Pybind11 and NumPy to expose GPU functions to Python.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdexcept> // Include for runtime_error

namespace py = pybind11;

// Forward declaration of the CUDA Host function from downsample_kernel.cu
// This tells the C++ compiler the function exists and where to find it during linking.
extern "C" void gpu_voxel_hash(
    const float* h_input_data, 
    unsigned int* h_output_hash, 
    int N, 
    float voxel_size
);

py::array_t<unsigned int> run_voxel_hash(py::array_t<float> input_points_py, float voxel_size) {
    // 1. Ensure input is a valid 2D array of floats (N, 4)
    py::buffer_info buf = input_points_py.request();
    
    if (buf.ndim != 2 || buf.shape[1] != 4) {
        throw std::runtime_error("Input array must be N x 4 (X, Y, Z, I).");
    }
    
    // Total number of points (N)
    int N = buf.shape[0];
    
    // Get raw pointer to the input data (Host memory)
    // We assume the NumPy array is contiguous in memory.
    const float* h_input_data = static_cast<const float*>(buf.ptr);

    // 2. Allocate Host memory for output hash keys
    std::vector<unsigned int> h_output_hash(N);

    // 3. Call the external CUDA Host function (defined and compiled by NVCC in the .cu file)
    gpu_voxel_hash(h_input_data, h_output_hash.data(), N, voxel_size);

    // 4. Return the result to Python as a NumPy array
    py::capsule free_when_done(h_output_hash.data(), [](void *f) {
        // Simple destructor placeholder
    });

    return py::array_t<unsigned int>(
        {N},                     // Shape of the output array
        {sizeof(unsigned int)},  // Stride (contiguous 1D array)
        h_output_hash.data(),    // Data pointer
        free_when_done           // Capsule to handle memory management
    );
}

// Pybind11 Module definition
PYBIND11_MODULE(slam_utils, m) {
    m.doc() = "Pybind11 module for GPU-accelerated SLAM utilities."; 

    // This line exposes 'run_voxel_hash' to Python as 'voxel_hash'
    m.def("voxel_hash", &run_voxel_hash, "Run GPU-accelerated voxel hashing on a point cloud.");
}