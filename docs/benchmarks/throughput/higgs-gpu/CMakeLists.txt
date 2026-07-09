cmake_minimum_required(VERSION 3.18)

project(opennn_training_speed_benchmarks LANGUAGES CXX)

if(NOT TARGET opennn_speed)
    add_executable(opennn_speed
        ${CMAKE_CURRENT_SOURCE_DIR}/opennn_speed.cpp)

    target_link_libraries(opennn_speed PRIVATE opennn)
endif()
