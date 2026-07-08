cmake_minimum_required(VERSION 4.0)
project(PoissonSolver
    VERSION 0.1
    LANGUAGES CXX CUDA Fortran
)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(FetchContent)

# Build for Voltas as well if possible
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
    set(CMAKE_CUDA_ARCHITECTURES 80 90 100)
else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90 100)
endif()

# All Fortran `.mod` files shall be placed into into a separate dir.
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

find_package(CUDAToolkit REQUIRED)

option(USE_MPI "Build distributed version" OFF)
if (USE_MPI)
    # `COMPONENTS` finds only the CXX and Fortran APIs
    find_package(MPI REQUIRED COMPONENTS CXX Fortran)

    FetchContent_Declare(
        heffte
        GIT_REPOSITORY https://github.com/mntjams/heffte
        GIT_TAG 818288f295cedcc025187322dfd5d1bcc49b1fac
    )

    # This variable is then read by HeFFTe's CMake.
    # `CACHE` vars persist between CMake runs.
    # `FORCE` writes to the variable even if e.g. user specified otherwise
    # with the `-D...` command line option.
    set(Heffte_ENABLE_CUDA ON CACHE BOOL "Enable heFFTe CUDA backend" FORCE)
    set(Heffte_ENABLE_GPU_AWARE_MPI ON CACHE BOOL "Enable heFFTe GPU aware MPI" FORCE)

    FetchContent_MakeAvailable(heffte)
else()
    FetchContent_Populate(
        vkfft
        GIT_REPOSITORY https://github.com/DTolm/VkFFT
        GIT_TAG v1.3.4
        SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/vkfft-src
        BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/vkfft-build
    )

    add_library(VkFFT INTERFACE)

    # Set CUDA (1) as backend
    target_compile_definitions(VkFFT INTERFACE VKFFT_BACKEND=1)

    # Tell VkFFT where CUDA Toolkit is
    target_compile_definitions(VkFFT INTERFACE CUDA_TOOLKIT_ROOT_DIR="${CUDAToolkit_TARGET_DIR}")

    # Have to use `SYSTEM` to prevent `VkFFT` warnings cluttering compilation output
    target_include_directories(VkFFT SYSTEM INTERFACE ${CMAKE_BINARY_DIR}/_deps/vkfft-src/vkFFT)
    target_link_libraries(VkFFT INTERFACE CUDA::cuda_driver CUDA::nvrtc)
endif()

# Define compiler flags.
add_library(common_flags INTERFACE)
target_compile_options(common_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wall -Wextra -Wno-unused-parameter
    -Wno-unused-variable>
    $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_subdirectory(src)
add_subdirectory(simulation)
add_subdirectory(tests/residuum)
add_subdirectory(tests/benchmarks)

# === Auto-generate .clangd ===
set(CLANGD_CONFIG
"CompileFlags:
  Add:
    - \"-I${CMAKE_SOURCE_DIR}/include\"
    - \"--no-cuda-version-check\"
    - \"-std=c++20\"
")

# Dynamically append the CUDA toolkit include directories
foreach(CUDA_INC IN LISTS CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    string(APPEND CLANGD_CONFIG "    - \"-I${CUDA_INC}\"\n")
endforeach()

string(APPEND CLANGD_CONFIG "
  Remove:
    - \"--expt-relaxed-constexpr\"
    - \"-Xcompiler=-fPIC\"
    - \"--generate-code*\"
    - \"--options-file\"
    - \"-forward-unknown-to-host-compiler\"
")

file(WRITE "${CMAKE_SOURCE_DIR}/.clangd" "${CLANGD_CONFIG}")
