cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(urinary_inflammations_diagnosis)

set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(urinary_inflammations_diagnosis main.cpp)

target_link_libraries(urinary_inflammations_diagnosis ${PROJECT_LINK_LIBS})

