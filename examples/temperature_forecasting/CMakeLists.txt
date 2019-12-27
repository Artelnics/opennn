cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(temperature_forecasting)

set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(temperature_forecasting main.cpp)

target_link_libraries(temperature_forecasting ${PROJECT_LINK_LIBS})

