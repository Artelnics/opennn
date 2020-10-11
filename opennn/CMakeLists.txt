

configure_file(opennn_config.h.in opennn_config.h)

file(GLOB opennn_src "*.cpp")
file(GLOB opennn_head "*.h")

add_library(opennn SHARED ${opennn_src})

target_include_directories(opennn PUBLIC
	 ${CMAKE_CURRENT_SOURCE_DIR}
	 "${PROJECT_BINARY_DIR}/opennn"
)

set_target_properties(opennn PROPERTIES PUBLIC_HEADER "${opennn_head}")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(opennn Threads::Threads OpenMP::OpenMP_CXX)



include(GNUInstallDirs)
install(TARGETS opennn
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opennn)

