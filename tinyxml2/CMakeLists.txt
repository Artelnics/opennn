cmake_minimum_required(VERSION 2.6)

add_library(tinyxml2 tinyxml2.h tinyxml2.cpp)

INSTALL(TARGETS tinyxml2 ARCHIVE DESTINATION lib INCLUDES DESTINATION include/tinyxml2)
INSTALL(DIRECTORY . DESTINATION include/tinyxml2 FILES_MATCHING PATTERN "*.h")
