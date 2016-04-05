
add_executable(logical_operations main.cpp)
target_link_libraries(logical_operations opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS logical_operations RUNTIME DESTINATION bin)