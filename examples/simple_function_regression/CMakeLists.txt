
add_executable(simple_function_regression main.cpp)
target_link_libraries(simple_function_regression opennn)

set(TEST_DATA_FILENAME simplefunctionregression.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS simple_function_regression RUNTIME DESTINATION bin)