# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(simple_pattern_recognition)

if(WIN32)
	if(EXISTS ../../opennn/Release/opennn)
		SET(PROJECT_LINK_LIBS ../../opennn/Release/opennn)
	else()
		SET(PROJECT_LINK_LIBS ../../opennn/Debug/opennn)
	endif()
endif(WIN32)

if(UNIX)
    if(EXISTS ../../opennn/Release/opennn)
		SET(PROJECT_LINK_LIBS ../../opennn/Release/opennn)
	else()
		SET(PROJECT_LINK_LIBS ../../opennn/Debug/opennn)
	endif()
endif(UNIX)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(simple_pattern_recognition main.cpp)

target_link_libraries(simple_pattern_recognition ${PROJECT_LINK_LIBS})
