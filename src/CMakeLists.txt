set(LIB_SRC_CXX
    c_binding.cpp
    detail/c_binding_for_fortran.cpp
    detail/runners/common.cu
    detail/runners/common_real.cu
    detail/runners/common_cmplx.cu
    detail/runners/nonuniform_z.cu
    mpi_utils.cpp
)
set(FORTRAN_BINDING
    fortran_binding.f90
)


# Debug files shall have a `_debug` suffix
set(CMAKE_DEBUG_POSTFIX "_debug")

# Output STATIC libraries here
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Output SHARED libraries here
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(poisson-solver ${LIB_SRC_CXX})

if (BUILD_FORTRAN)
    target_sources(poisson-solver PRIVATE ${FORTRAN_BINDING})
endif()

# The `BUILD_INTERFACE` evals to true when target 
# linking againts this lib is part of the same CMake build tree
# (e.g. called `add_subdirectory(libs/poisson-solver))`.
target_include_directories(poisson-solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
)

target_link_libraries(poisson-solver PRIVATE common_flags)
target_link_libraries(poisson-solver PUBLIC CUDA::cudart VkFFT)
if (USE_MPI)
    target_link_libraries(poisson-solver PUBLIC
        Heffte
        MPI::MPI_CXX
    )
    if (BUILD_FORTRAN)
        target_link_libraries(poisson-solver PUBLIC
            MPI::MPI_Fortran
        )
    endif()

    # Defines `USE_MPI`, therefore code can do `#ifdef USE_MPI`
    target_compile_definitions(poisson-solver PRIVATE USE_MPI)
endif()
