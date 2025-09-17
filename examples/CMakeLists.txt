cmake_minimum_required(VERSION 3.18)

if(UNIX)
	option(USE_OpenMP "Use OpenMP" ON)
	if(USE_OpenMP)
		find_package(OpenMP)
		if(OPENMP_FOUND)
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		endif()
	endif()
endif()

project(examples)

add_subdirectory(airfoil_self_noise)
add_subdirectory(amazon_reviews)
add_subdirectory(breast_cancer)
add_subdirectory(emotion_analysis)
add_subdirectory(forecasting)
add_subdirectory(iris_plant)
add_subdirectory(mnist)
add_subdirectory(translation)


