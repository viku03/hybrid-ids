#!/bin/bash
# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set Homebrew LLVM as default compiler
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
# Add OpenMP paths
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export CMAKE_PREFIX_PATH="$(brew --prefix pytorch)"

# Create build directory
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

# Parallel Version
echo "Compiling Parallel NIDS..."
mkdir -p parallel
cd parallel
cmake ../.. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
-DBUILD_PARALLEL=ON
cmake --build . --config Release -- -j$(sysctl -n hw.ncpu)

# Run Parallel NIDS and capture output
echo "Running Parallel NIDS..."
"${SCRIPT_DIR}/parallel_nids" > "${SCRIPT_DIR}/build/nids_benchmark_results.txt" 2>&1

# Return to build directory
cd ..

# Sequential Version
echo "Compiling Sequential NIDS..."
mkdir -p sequential
cd sequential
cmake ../.. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
-DBUILD_PARALLEL=OFF
cmake --build . --config Release -- -j$(sysctl -n hw.ncpu)

# Run Sequential NIDS and capture output
echo "Running Sequential NIDS..."
"${SCRIPT_DIR}/sequential_nids" > "${SCRIPT_DIR}/build/sequential_nids_benchmark_results.txt" 2>&1

# Return to build directory
cd ..

# Visualize Results (if Python script exists)
if [ -f "${SCRIPT_DIR}/compare_performance.py" ]; then
    echo "Generating Performance Visualization..."
    python3 "${SCRIPT_DIR}/compare_performance.py"
fi

# Display benchmark results
echo -e "\nParallel NIDS Benchmark Results:"
cat "${SCRIPT_DIR}/build/nids_benchmark_results.txt"

echo -e "\nSequential NIDS Benchmark Results:"
cat "${SCRIPT_DIR}/build/sequential_nids_benchmark_results.txt"

# Exit with error if either compilation failed
if [ $? -ne 0 ]; then
    exit 1
fi