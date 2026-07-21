cmake_minimum_required(VERSION 3.18)

project(emotion_analysis)

add_executable(emotion_analysis main.cpp)

add_custom_command(TARGET emotion_analysis POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/emotion_analysis"
)

target_link_libraries(emotion_analysis PUBLIC opennn)
