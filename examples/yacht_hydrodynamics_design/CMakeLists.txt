cmake_minimum_required(VERSION 2.8.10)

# Project's name

project(yacht_hydrodynamics_design)

link_directories(${CMAKE_SOURCE_DIR}/opennn)

include_directories(${CMAKE_SOURCE_DIR}/opennn)

add_executable(yacht_hydrodynamics_design main.cpp)

target_link_libraries(yacht_hydrodynamics_design opennn)

set (source "${CMAKE_CURRENT_SOURCE_DIR}/data")
set (destination "${CMAKE_BINARY_DIR}/examples/data")

file(GLOB MY_PUBLIC_HEADERS
  "${source}/*.*"
)
file(COPY ${MY_PUBLIC_HEADERS} DESTINATION "${destination}")