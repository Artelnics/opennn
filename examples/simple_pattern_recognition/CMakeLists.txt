add_executable(simple_pattern_recognition main.cpp)
target_link_libraries(simple_pattern_recognition opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS simple_pattern_recognition RUNTIME DESTINATION bin)