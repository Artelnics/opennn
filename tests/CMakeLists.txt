cmake_minimum_required(VERSION 2.8.12)

project(tests)

if(WIN32)
	set(PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif()

if(UNIX)
    if(CMAKE_GENERATOR MATCHES "Visual Studio 16 2019")
		set(PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/Release/opennn.lib)
	else()
		set(PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/build/opennn/libopennn.a)
	endif()
endif()

file(GLOB SOURCES "*.cpp")

add_executable(tests ${SOURCES})

target_link_libraries(tests PUBLIC opennn)
