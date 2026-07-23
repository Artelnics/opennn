cmake_minimum_required(VERSION 3.18)

project(dipcan_metastasis)

if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(dipcan_metastasis main.cpp)

target_link_libraries(dipcan_metastasis PUBLIC opennn)
