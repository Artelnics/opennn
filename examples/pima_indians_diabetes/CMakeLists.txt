
add_executable(pima_indians_diabetes main.cpp)
target_link_libraries(pima_indians_diabetes opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS pima_indians_diabetes RUNTIME DESTINATION bin)