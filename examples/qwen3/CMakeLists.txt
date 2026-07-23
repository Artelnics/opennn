cmake_minimum_required(VERSION 3.18)

project(qwen3)

if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/Release/opennn/libopennn.a)
endif()

if(WIN32)
	set(PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif()

add_executable(qwen3 main.cpp)

target_link_libraries(qwen3 PUBLIC opennn)
