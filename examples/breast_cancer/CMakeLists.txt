
add_executable(breast_cancer main.cpp)
target_link_libraries(breast_cancer opennn)

set(TEST_DATA_FILENAME breast_cancer.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS breast_cancer RUNTIME DESTINATION bin)