
add_executable(yacht_hydrodynamics_design main.cpp)
target_link_libraries(yacht_hydrodynamics_design opennn)

set(TEST_DATA_FILENAME yachtresistance.dat)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS yacht_hydrodynamics_design RUNTIME DESTINATION bin)