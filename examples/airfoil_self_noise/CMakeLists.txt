
add_executable(airfoil_self_noise main.cpp)
target_link_libraries(airfoil_self_noise opennn)

set(TEST_DATA_FILENAME airfoil_self_noise.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS airfoil_self_noise RUNTIME DESTINATION bin)