
add_executable(iris_plant main.cpp)
target_link_libraries(iris_plant opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS iris_plant RUNTIME DESTINATION bin)