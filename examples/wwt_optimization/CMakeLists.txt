# examples/wwt_optimization/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

project(wwt_optimization)

if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/Release/opennn/libopennn.a)
endif()

if(WIN32)
	set(PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif()

add_executable(wwt_optimization main.cpp)

add_custom_command(TARGET wwt_optimization POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data"
)

target_link_libraries(wwt_optimization PUBLIC opennn)
