set(LIB_SRC
    c_binding.cpp
    fortran_binding.f90
    kernels/runners.cu
)

# Debug files shall have a `_debug` suffix
set(CMAKE_DEBUG_POSTFIX "_debug")

# Output STATIC libraries here
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Output SHARED libraries here
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Note that `<SHARED|STATIC>` can be specified
# by using `-DBUILD_SHARED_LIBS=<Yes|No>`.
# The original PoisFFT outputted a shared lib.
add_library(poisson-solver ${LIB_SRC})

# The `BUILD_INTERFACE` evals to true when target 
# linking againts this lib is part of the same CMake build tree
# (e.g. called `add_subdirectory(libs/poisson-solver))`.
target_include_directories(poisson-solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>)

target_link_libraries(poisson-solver PRIVATE common_flags)
target_link_libraries(poisson-solver PUBLIC CUDA::cufft CUDA::cudart)
