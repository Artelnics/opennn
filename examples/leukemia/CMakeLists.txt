
add_executable(leukemia main.cpp)
target_link_libraries(leukemia opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS leukemia RUNTIME DESTINATION bin)