# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.12)

# Project's name

project(logical_operations)
IF(UNIX)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/Release/opennn/libopennn.a)
ENDIF()

if (WIN32)
set (PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif()


link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(logical_operations main.cpp)

target_link_libraries(logical_operations ${PROJECT_LINK_LIBS})

