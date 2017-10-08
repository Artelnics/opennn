
add_library(tinyxml2 tinyxml2.h tinyxml2.cpp)

set_target_properties(tinyxml2 PROPERTIES PUBLIC_HEADER "tinyxml2.h")
install(TARGETS tinyxml2 
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/tinyxml2
)