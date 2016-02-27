
set(BLANK_SRCS
        main.cpp)

add_executable(blank ${BLANK_SRCS})
target_link_libraries(blank opennn)