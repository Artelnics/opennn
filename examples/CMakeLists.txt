# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8.12)

IF(UNIX)
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
ENDIF()
ENDIF()


# Project's name
project(examples)
add_subdirectory(airfoil_self_noise)
add_subdirectory(breast_cancer)
add_subdirectory(iris_plant)
add_subdirectory(leukemia)
add_subdirectory(logical_operations)
add_subdirectory(mnist)
add_subdirectory(pima_indians_diabetes)
add_subdirectory(rosenbrock)
add_subdirectory(urinary_inflammations_diagnosis)
add_subdirectory(yacht_hydrodynamics_design)
add_subdirectory(yacht_hydrodynamics_production)
