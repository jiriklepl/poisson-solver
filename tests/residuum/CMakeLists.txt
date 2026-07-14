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
if (USE_MPI)
    #TODO - Add sources for CXX MPI tests
    set(CXX_TEST_SRC "")
else()
    set(CXX_TEST_SRC
        test_main.cpp
        test_residuum.cpp
        test_residuum_nonuniform_z.cpp
    )
endif()

# TODO - Remove "if" statement when the CXX_TEST_SRC is
# guaranteed to be always populated (i.e., when MPI tests are rdy)
if (CXX_TEST_SRC)
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

    # TODO - MPI test launching via srun is unhandled here

    # Register residuum tests
    doctest_discover_tests(residuum-tests)
endif()

# == Fortran Residuum Tests ==
if (USE_MPI)
    set(FORTRAN_TEST_SRCS
        test_residuum_mpi.f90
    )
else()
    set(FORTRAN_TEST_SRCS
        test_residuum.f90
        test_residuum_nonuniform_z.f90
    )
endif()

set(ALL_FORTRAN_TEST_TARGETS "")

foreach(f_test_src IN LISTS FORTRAN_TEST_SRCS)
    # Extract filename
    get_filename_component(base_name ${f_test_src} NAME_WE)

    set(test_target "fortran-${base_name}")

    # By default, only the lib is built
    add_executable(${test_target} EXCLUDE_FROM_ALL ${f_test_src})

    target_include_directories(${test_target} PRIVATE
        ${CMAKE_Fortran_MODULE_DIRECTORY})

    # Automatically links with `stdc++`
    target_link_libraries(${test_target} PRIVATE
        common_flags
        poisson-solver
    )

    # Register test with ctest
    if (USE_MPI)
        add_test(
            NAME ${test_target}
            COMMAND srun
            --mpi=pmix
            -N 2
            --ntasks-per-node 2
            -p gpu-short-teach
            -w "volta[02-05]"
            --gpus-per-task 1
            ch-run
            -b ../../../../:/opt/build
            ../../../../../imgdir --
            $<TARGET_FILE:${test_target}>
        )
        set_tests_properties(${test_target} PROPERTIES PROCESSORS 4)
    else()
        add_test(
            NAME ${test_target}
            COMMAND ${test_target}
        )
    endif()

    # Tests fail when `FAIL` appears on output
    set_tests_properties(${test_target}
        PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL"
    )

    list(APPEND ALL_FORTRAN_TEST_TARGETS ${test_target})
endforeach()

# == Group Tests Together ==
add_custom_target(tests)

# TODO - When CXX MPI tests are rdy, keep only the else branch
if (USE_MPI)
    add_dependencies(tests ${ALL_FORTRAN_TEST_TARGETS})
else()
    add_dependencies(tests residuum-tests ${ALL_FORTRAN_TEST_TARGETS})
endif()
