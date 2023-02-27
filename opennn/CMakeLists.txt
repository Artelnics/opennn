cmake_minimum_required(VERSION 2.8.12)

project(opennn)

file(GLOB SOURCES *.cpp)

add_library(opennn ${SOURCES})
