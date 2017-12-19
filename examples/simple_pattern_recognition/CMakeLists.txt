add_executable(simple_pattern_recognition main.cpp)
target_link_libraries(simple_pattern_recognition opennn)

set(TEST_DATA_FILENAME simple_pattern_recognition.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS simple_pattern_recognition RUNTIME DESTINATION bin)