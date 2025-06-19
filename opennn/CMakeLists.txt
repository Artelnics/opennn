project(opennn_library)

file(GLOB_RECURSE SOURCES "*.cpp" "*.h")

add_library(opennn ${SOURCES})

target_include_directories(opennn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})