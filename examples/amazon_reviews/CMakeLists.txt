cmake_minimum_required(VERSION 3.18)

project(amazon_reviews)

add_executable(amazon_reviews main.cpp)

add_custom_command(TARGET amazon_reviews POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "${CMAKE_BINARY_DIR}/data/amazon_reviews"
)


target_link_libraries(amazon_reviews PRIVATE opennn)
