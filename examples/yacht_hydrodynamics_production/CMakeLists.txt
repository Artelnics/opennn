
add_executable(yacht_hydrodynamics_production main.cpp)
target_link_libraries(yacht_hydrodynamics_production opennn)

set(TEST_DATA_FILENAME neural_network.xml)
configure_file(data/${TEST_DATA_FILENAME} ${CURRENT_TEST_DATA_FOLDER}/${TEST_DATA_FILENAME} COPYONLY)

install(TARGETS yacht_hydrodynamics_production RUNTIME DESTINATION bin)