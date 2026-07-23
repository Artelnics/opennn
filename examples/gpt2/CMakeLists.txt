cmake_minimum_required(VERSION 3.18)

project(gpt2)

add_executable(gpt2 main.cpp)

target_link_libraries(gpt2 PRIVATE opennn)
