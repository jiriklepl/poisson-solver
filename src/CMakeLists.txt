set(LIB_SRC
    c_binding.cpp
    detail/c_binding_for_fortran.cpp
    fortran_binding.f90
    detail/runners/common.cu
    detail/runners/common_real.cu
    detail/runners/common_cmplx.cu
    detail/runners/nonuniform_z.cu
    mpi_utils.cpp
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
)

target_link_libraries(poisson-solver PRIVATE common_flags VkFFT)
target_link_libraries(poisson-solver PUBLIC CUDA::cudart)
if (USE_MPI)
    target_link_libraries(poisson-solver PUBLIC
        MPI::MPI_CXX
        MPI::MPI_Fortran
    )

    target_link_libraries(poisson-solver PRIVATE Heffte)

    # Defines `USE_MPI`, therefore code can do `#ifdef USE_MPI`
    target_compile_definitions(poisson-solver PUBLIC USE_MPI)
endif()
