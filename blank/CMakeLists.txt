cmake_minimum_required(VERSION 3.18)

project(blank)

add_executable(blank main.cpp)

if(UNIX AND OpenMP_FOUND)
    if(NOT CMAKE_CXX_FLAGS MATCHES "${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

if(WIN32)
	set(PROJECT_LINK_LIBS ../opennn/Release/opennn)
endif()

target_include_directories(blank PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../opennn
)

target_link_libraries(blank PUBLIC opennn)






