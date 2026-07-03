cmake_minimum_required(VERSION 3.18)

project(recurrent_lstm_forecasting_benchmark)

if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(PROJECT_LINK_LIBS ${CMAKE_SOURCE_DIR}/Release/opennn/libopennn.a)
endif()

if(WIN32)
	set(PROJECT_LINK_LIBS ../../opennn/Release/opennn)
endif()

add_executable(recurrent_lstm_forecasting_benchmark main.cpp)

target_link_libraries(recurrent_lstm_forecasting_benchmark PUBLIC opennn)
