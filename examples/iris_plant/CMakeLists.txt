
add_executable(iris_plant main.cpp)
target_link_libraries(iris_plant opennn)

set(TEST_DATA_FILENAME iris_plant.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS iris_plant RUNTIME DESTINATION bin)