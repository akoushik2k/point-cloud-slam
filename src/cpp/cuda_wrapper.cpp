/**
 * @file cuda_wrapper.cpp
 * @brief C++ wrapper to call the CUDA kernel and verify results.
 */
#include <iostream>
#include <vector>

// Forward declaration of the external CUDA function defined in simple_kernel.cu
extern "C" void run_cuda_test(float *h_A, float *h_B, float *h_C, int N);

int main(int argc, char** argv) {
    std::cout << "--- CUDA Test Program (Vector Addition) ---" << std::endl;

    const int N = 1024 * 1024; // 1 million elements
    std::cout << "Running vector addition on " << N << " elements..." << std::endl;
    
    // Allocate Host memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    // Initialize Host input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Call the CUDA function (which runs on the RTX 4060)
    run_cuda_test(h_A.data(), h_B.data(), h_C.data(), N);
    
    // Verification: Check a few elements
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "SUCCESS: CUDA kernel execution confirmed (1.0 + 2.0 = 3.0)." << std::endl;
    } else {
        std::cerr << "FAILURE: CUDA kernel output verification failed." << std::endl;
    }

    return 0;
}