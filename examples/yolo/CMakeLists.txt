cmake_minimum_required(VERSION 3.18)

project(yolo)

add_executable(yolo main.cpp)

target_link_libraries(yolo PRIVATE opennn)

# REUSE_FROM requires identical compile flags between targets; that holds on
# MSVC but breaks CMake generation with Makefiles/GCC (PCH path unresolved),
# so non-MSVC builds use their own PCH from the same header.
if(MSVC)
    target_precompile_headers(yolo REUSE_FROM opennn)
else()
    target_precompile_headers(yolo PRIVATE ${CMAKE_SOURCE_DIR}/opennn/pch.h)
endif()

