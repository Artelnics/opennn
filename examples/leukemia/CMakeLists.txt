
add_executable(leukemia main.cpp)
target_link_libraries(leukemia opennn)

set(TEST_DATA_FILENAME leukemia.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS leukemia RUNTIME DESTINATION bin)