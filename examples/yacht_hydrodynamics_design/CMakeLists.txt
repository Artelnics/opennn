cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(yacht_hydrodynamics_design)

set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(yacht_hydrodynamics_design main.cpp)

target_link_libraries(yacht_hydrodynamics_design ${PROJECT_LINK_LIBS})

