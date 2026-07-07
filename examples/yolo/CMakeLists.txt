cmake_minimum_required(VERSION 3.18)

project(yolo)

if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/Release/opennn/libopennn.a)
endif()

if(WIN32)
	set(PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif()

add_executable(yolo main.cpp)

target_link_libraries(yolo PUBLIC opennn)

# REUSE_FROM requires identical compile flags between targets; that holds on
# MSVC but breaks CMake generation with Makefiles/GCC (PCH path unresolved),
# so non-MSVC builds use their own PCH from the same header.
if(MSVC)
    target_precompile_headers(yolo REUSE_FROM opennn)
else()
    target_precompile_headers(yolo PRIVATE ${CMAKE_SOURCE_DIR}/opennn/pch.h)
endif()

