cmake_minimum_required(VERSION 2.8.12)

project(opennn)

set(CMAKE_CXX_STANDARD 11)
set(CPACK_GENERATOR "TGZ")

set(CMAKE_SUPPRESS_REGENERATION true)
set(VS_STARTUP_PROJECT opennn)

#if(__OPENNN_OMP__)
#    message("Using OpenMP")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")#${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")#${OpenMP_CXX_FLAGS}")
#    set(CMAEK_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# Uncomment next line to compile without using C++11
# add_definitions(-D__Cpp11__)

add_subdirectory(opennn)

include_directories(opennn)

option(OpenNN_BUILD_EXAMPLES "Build OpenNN examples" ON)

option(OpenNN_BUILD_BLANK "Build OpenNN blank" ON)

option(OpenNN_BUILD_TESTS "Build OpenNN tests" ON)

if(OpenNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(OpenNN_BUILD_EXAMPLES)

if(OpenNN_BUILD_BLANK)
    add_subdirectory(blank)
endif(OpenNN_BUILD_BLANK)

if(OpenNN_BUILD_TESTS)
    add_subdirectory(tests)
endif(OpenNN_BUILD_TESTS)

include(CPack)