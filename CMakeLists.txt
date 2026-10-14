cmake_minimum_required(VERSION 4.0)
project(PoissonSolver
    VERSION 0.1
    LANGUAGES CXX
)

find_package(CUDAToolkit REQUIRED)

if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX}
        AND CUDAToolkit_NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()

# Build for Voltas as well if possible
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    set(CMAKE_CUDA_ARCHITECTURES 80 90 100)
else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90 100)
endif()

enable_language(CUDA)

option(POISSON_SOLVER_BUILD_FORTRAN "Build Fortran binding" ON)

if (POISSON_SOLVER_BUILD_FORTRAN)
    enable_language(Fortran)

    # All Fortran `.mod` files shall be placed into into a separate dir.
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
endif()

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(FetchContent)

option(POISSON_SOLVER_USE_MPI "Build distributed version" OFF)
if (POISSON_SOLVER_USE_MPI)
    # `COMPONENTS` finds only the CXX and (maybe) Fortran APIs
    find_package(MPI REQUIRED COMPONENTS CXX)
    if (POISSON_SOLVER_BUILD_FORTRAN)
        find_package(MPI REQUIRED COMPONENTS Fortran)
    endif()

    FetchContent_Declare(
        heffte
        GIT_REPOSITORY https://github.com/mntjams/heffte
        GIT_TAG 38f9b9b546561d2b2a172afc2168437ae7426889
    )

    # This variable is then read by HeFFTe's CMake.
    # `CACHE` vars persist between CMake runs.
    # `FORCE` writes to the variable even if e.g. user specified otherwise
    # with the `-D...` command line option.
    set(Heffte_ENABLE_CUDA ON CACHE BOOL "Enable heFFTe CUDA backend" FORCE)
    set(Heffte_ENABLE_GPU_AWARE_MPI ON CACHE BOOL "Enable heFFTe GPU aware MPI" FORCE)

    FetchContent_MakeAvailable(heffte)
endif()

option(POISSON_SOLVER_ENABLE_BENCHMARKS
    "Enable measurements and the benchmarks target" OFF)

# VkFFT is required even for MPI builds
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

# Define compiler flags.
add_library(common_flags INTERFACE)
target_compile_options(common_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX,CUDA>:-Wall -Wextra -Wno-unused-parameter
    -Wno-unused-variable>
    $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:Fortran>:-cpp>
)

add_subdirectory(src)
add_subdirectory(tests/residuum)
if (POISSON_SOLVER_ENABLE_BENCHMARKS)
    add_subdirectory(tests/benchmarks)
endif()
add_subdirectory(examples)

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

# === Doxygen Documentation ===
find_package(Doxygen)
if (DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME "poisson-solver")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)

    if (POISSON_SOLVER_USE_MPI)
        set(DOXYGEN_PREDEFINED "POISSON_SOLVER_USE_MPI")
    endif()

    # Make docs prettier
    FetchContent_Declare(
        doxygen_awesome_css
        GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
        GIT_TAG v2.3.4
    )
    FetchContent_MakeAvailable(doxygen_awesome_css)

    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET
        "${doxygen_awesome_css_SOURCE_DIR}/doxygen-awesome.css"
        "${doxygen_awesome_css_SOURCE_DIR}/doxygen-awesome-sidebar-only.css"
    )

    doxygen_add_docs(doc
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
endif()
