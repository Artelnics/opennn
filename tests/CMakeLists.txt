# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(tests)

if (WIN32)
    if (CMAKE_GENERATOR MATCHES "Visual Studio 15 2017")
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/Release/opennn.lib)
	else()
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/libopennn.a)
	endif()
endif (WIN32)

if (UNIX)
    if (CMAKE_GENERATOR MATCHES "Visual Studio 15 2017")
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/Release/opennn.lib)
	else()
		set (PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/libopennn.a)
	endif()
endif (UNIX)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/tests)

file(GLOB SOURCES "*.cpp")

add_executable(tests ${SOURCES})

target_link_libraries(tests ${PROJECT_LINK_LIBS})
