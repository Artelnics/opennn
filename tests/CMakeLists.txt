# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

set(PROJECT_NAME tests)
project(${PROJECT_NAME})

if (WIN32)
    set (PROJECT_LINK_LIBS opennn)
endif (WIN32)

if (UNIX)
    set (PROJECT_LINK_LIBS opennn)
endif (UNIX)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

file(GLOB SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${PROJECT_LINK_LIBS})
