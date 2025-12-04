cmake_minimum_required(VERSION 3.10)
project(PointCloudSLAM LANGUAGES CXX CUDA)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Find PCL (Point Cloud Library) ---
# Ensure PCL headers and libraries are found
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# --- 2. C++ PCL Test Executable ---
# A simple executable to check PCL linking and basic C++ compilation
add_executable(pcl_test src/cpp/main.cpp)

# Enable OpenMP for parallel CPU processing (good for PCL)
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    # Apply options AFTER the target 'pcl_test' is created.
    target_link_options(pcl_test PUBLIC ${OpenMP_CXX_FLAGS})
    target_compile_options(pcl_test PUBLIC ${OpenMP_CXX_FLAGS})
endif()

# Link the necessary PCL components using the general variable ${PCL_LIBRARIES} 
target_link_libraries(pcl_test ${PCL_LIBRARIES})

# --- 3. CUDA Configuration ---
set(CUDA_SEPARABLE_COMPILATION ON)
cmake_policy(SET CMP0104 NEW) # Silence the architecture warning

# FIX: Reverting to 8.6 to resolve the 'Unsupported gpu architecture' error, 
# likely due to an older NVCC (e.g., 11.x) being found in the path despite CUDA 12.4 runtime.
set(COMMON_CUDA_ARCH "86") 

# --- 4. CUDA Test Executable (Keep for reference/testing) ---
add_executable(cuda_test src/cuda/simple_kernel.cu src/cpp/cuda_wrapper.cpp)
target_link_libraries(cuda_test ${CUDA_LIBRARIES})

# Apply modern CUDA properties to cuda_test
target_compile_features(cuda_test PUBLIC cxx_std_17)
set_property(TARGET cuda_test PROPERTY CUDA_ARCHITECTURES ${COMMON_CUDA_ARCH})
target_compile_options(cuda_test PRIVATE -std=c++17)


# --- 5. Pybind11 Setup for Python Integration (NEW SECTION) ---
# Pybind11 is used to create Python bindings for the C++ code.
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# The Pybind11 module target is a shared library (.so file)
pybind11_add_module(slam_utils MODULE src/cpp/gpu_processing.cpp src/cuda/downsample_kernel.cu)

# This is the path where Python will look for your module.
set_target_properties(slam_utils PROPERTIES SUFFIX ".so") 

# Apply modern CUDA properties to slam_utils
target_compile_features(slam_utils PUBLIC cxx_std_17)
set_property(TARGET slam_utils PROPERTY CUDA_ARCHITECTURES ${COMMON_CUDA_ARCH})
target_compile_options(slam_utils PRIVATE -std=c++17)

# FIX: Use the 'PUBLIC' keyword to ensure the link signature matches the one
# used internally by pybind11_add_module, resolving the conflict.
target_link_libraries(slam_utils 
    PUBLIC # Explicitly use a keyword
    ${PCL_LIBRARIES} 
    ${CUDA_LIBRARIES}
)