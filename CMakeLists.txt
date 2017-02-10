
cmake_minimum_required(VERSION 2.6)
project(OpenNN)

set (CMAKE_CXX_STANDARD 11)
SET(CPACK_GENERATOR "TGZ")

if(__OPENNN_MPI__)
   find_package(MPI)
   if(MPI_FOUND)
        message("Using MPI")
        set(CMAKE_CXX_COMPILER mpicxx)
        set(CMAKE_C_COMPILER mpicc)
        set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
        set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
        add_definitions(-D__OPENNN_MPI__ )
    endif()
endif()


if(__OPENNN_OMP__)
    message("Using OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")#${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")#${OpenMP_CXX_FLAGS}")
    set(CMAEK_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Uncomment next line to compile without using C++11
#add_definitions(-D__Cpp11__)

add_subdirectory(tinyxml2)
add_subdirectory(opennn)
include_directories(opennn)
add_subdirectory(examples)
add_subdirectory(blank)
add_subdirectory(tests)

include(CPack)
