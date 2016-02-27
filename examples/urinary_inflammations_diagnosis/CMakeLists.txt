
add_executable(urinary_inflammations_diagnosis main.cpp)
target_link_libraries(urinary_inflammations_diagnosis opennn)

install(DIRECTORY data/ DESTINATION data/)

install(TARGETS urinary_inflammations_diagnosis RUNTIME DESTINATION bin)