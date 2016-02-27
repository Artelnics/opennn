
add_executable(simple_function_regression main.cpp)
target_link_libraries(simple_function_regression opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS simple_function_regression RUNTIME DESTINATION bin)