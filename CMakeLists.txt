cmake_minimum_required(VERSION 3.10)
project(PointCloudSLAM LANGUAGES CXX CUDA)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Find PCL (Point Cloud Library) ---
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

# --- 2. Find CUDA ---
# The CUDA package finds your installed CUDA toolkit (version 12.4)
find_package(CUDA REQUIRED)

# Define an executable (e.g., for testing C++ PCL functions)
add_executable(pcl_test src/cpp/main.cpp)

# Link PCL libraries to your executable
target_link_libraries(pcl_test ${PCL_LIBRARIES})

# --- 3. Example for CUDA Compilation ---
# Let's create a CUDA-accelerated library or executable later
# cuda_add_executable(cuda_test src/cuda/kernel.cu)
# target_link_libraries(cuda_test ${PCL_LIBRARIES})