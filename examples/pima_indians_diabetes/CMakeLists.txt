
add_executable(pima_indians_diabetes main.cpp)
target_link_libraries(pima_indians_diabetes opennn)

set(TEST_DATA_FILENAME pima_indians_diabetes.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS pima_indians_diabetes RUNTIME DESTINATION bin)