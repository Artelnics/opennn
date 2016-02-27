
add_executable(breast_cancer main.cpp)
target_link_libraries(breast_cancer opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS breast_cancer RUNTIME DESTINATION bin)