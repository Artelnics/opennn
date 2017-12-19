
add_executable(logical_operations main.cpp)
target_link_libraries(logical_operations opennn)

set(TEST_DATA_FILENAME logical_operations.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS logical_operations RUNTIME DESTINATION bin)