set(SIM_SRC
    simulation.cpp
)

# By default, only the lib is built
add_executable(simulation EXCLUDE_FROM_ALL ${SIM_SRC})

target_link_libraries(simulation PRIVATE common_flags poisson-solver)

