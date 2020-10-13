cmake_minimum_required(VERSION 2.8.10)

project(example DESCRIPTION "Usage example for OpenNN shared library")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)


add_subdirectory(airfoil_self_noise)
add_subdirectory(breast_cancer)
add_subdirectory(iris_plant)
add_subdirectory(leukemia)
add_subdirectory(logical_operations)
add_subdirectory(mnist)
add_subdirectory(pima_indians_diabetes)
add_subdirectory(rosenbrock)
add_subdirectory(simple_approximation)
add_subdirectory(simple_classification)
add_subdirectory(urinary_inflammations_diagnosis)
add_subdirectory(yacht_hydrodynamics_design)
add_subdirectory(yacht_hydrodynamics_production)