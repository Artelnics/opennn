
add_executable(yacht_hydrodynamics_design main.cpp)
target_link_libraries(yacht_hydrodynamics_design opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS yacht_hydrodynamics_design RUNTIME DESTINATION bin)