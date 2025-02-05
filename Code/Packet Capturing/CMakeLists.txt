cmake_minimum_required(VERSION 3.18)
project(nids CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to toggle between parallel and sequential builds
option(BUILD_PARALLEL "Build Parallel NIDS version" OFF)

# Use Homebrew's LLVM on macOS
if(APPLE)
    set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
    set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)
endif()

# Find packages
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Apple Silicon Optimization
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Suppress the kineto warning
if(TORCH_FOUND)
    list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "kineto_LIBRARY-NOTFOUND")
endif()

# Create executable based on build type
if(BUILD_PARALLEL)
    add_executable(parallel_nids Parallel_nids.cpp)
    target_link_libraries(parallel_nids
        PRIVATE
        ${TORCH_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
    set_target_properties(parallel_nids PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
else()
    add_executable(sequential_nids sequential_nids.cpp)
    target_link_libraries(sequential_nids
        PRIVATE
        ${TORCH_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
    set_target_properties(sequential_nids PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()