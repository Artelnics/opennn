cmake_minimum_required(VERSION 3.18)

project(forecasting_tinyml)

add_executable(forecasting_tinyml main.cpp)

target_link_libraries(forecasting_tinyml PUBLIC opennn)
