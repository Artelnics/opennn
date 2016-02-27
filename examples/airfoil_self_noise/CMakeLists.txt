
add_executable(airfoil_self_noise main.cpp)
target_link_libraries(airfoil_self_noise opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS airfoil_self_noise RUNTIME DESTINATION bin)