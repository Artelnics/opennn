
add_executable(urinary_inflammations_diagnosis main.cpp)
target_link_libraries(urinary_inflammations_diagnosis opennn)

set(TEST_DATA_FILENAME urinary_inflammations_diagnosis.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS urinary_inflammations_diagnosis RUNTIME DESTINATION bin)