# == C/C++ Residuum Tests ==
set(CXX_TEST_SRC
    test_residuum.cpp
)

# By default, only the lib is built
add_executable(residuum-tests EXCLUDE_FROM_ALL ${CXX_TEST_SRC})

target_include_directories(residuum-tests PRIVATE ../include/)
target_link_libraries(residuum-tests PRIVATE common_flags poisson-solver)

# == Fortran Residuum Tests ==
set(FORTRAN_TEST_SRC
    test_residuum.f90
)

# By default, only the lib is built
add_executable(fortran-residuum-tests EXCLUDE_FROM_ALL ${FORTRAN_TEST_SRC})
target_include_directories(fortran-residuum-tests PRIVATE 
    ${CMAKE_Fortran_MODULE_DIRECTORY})

# Automatically links with `stdc++`
target_link_libraries(fortran-residuum-tests PRIVATE common_flags
    poisson-solver)

