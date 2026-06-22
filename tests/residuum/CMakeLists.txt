# Fetch doctest
include(FetchContent)
set(DOCTEST_VERSION v2.5.2)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest
    GIT_TAG v2.5.2
)
FetchContent_MakeAvailable(doctest)

set(DOCTEST_INCLUDE_DIR ${doctest_SOURCE_DIR})

# Include doctest_discover_tests(<target>) function
include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

# == C/C++ Residuum Tests ==
set(CXX_TEST_SRC
    test_main.cpp
    test_residuum.cpp
    test_residuum_nonuniform_z.cpp
)

# By default, only the lib is built
add_executable(residuum-tests EXCLUDE_FROM_ALL ${CXX_TEST_SRC})

target_include_directories(residuum-tests PRIVATE
    ../include/
)
target_link_libraries(residuum-tests PRIVATE
    common_flags
    poisson-solver
    doctest::doctest
)

# Register residuum tests
doctest_discover_tests(residuum-tests)

# == Fortran Residuum Tests ==
set(FORTRAN_TEST_SRC
    test_residuum.f90
)

# By default, only the lib is built
add_executable(fortran-residuum-tests EXCLUDE_FROM_ALL ${FORTRAN_TEST_SRC})
target_include_directories(fortran-residuum-tests PRIVATE 
    ${CMAKE_Fortran_MODULE_DIRECTORY})

# Automatically links with `stdc++`
target_link_libraries(fortran-residuum-tests PRIVATE
    common_flags
    poisson-solver
)

# Register tests with ctest
add_test(
    NAME fortran-residuum-suite
    COMMAND fortran-residuum-tests
)

# Tests fail when `FAIL` appears on output
set_tests_properties(fortran-residuum-suite
    PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL"
)

# == Group Tests Together ==
add_custom_target(tests)
add_dependencies(tests residuum-tests fortran-residuum-tests)

