cmake_minimum_required(VERSION 3.18)

project(opennn_library LANGUAGES CXX)

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "*.h")

if(OpenNN_SHARED_LIBS)
    add_library(opennn SHARED
        ${CPP_SOURCES}
        ${HEADER_FILES}
    )
else()
    add_library(opennn STATIC 
        ${CPP_SOURCES} 
        ${HEADER_FILES}
    )
endif()

if(NOT OpenNN_SHARED_LIBS OR OpenNN_FORCE_WHOLE_ARCHIVE)
    # Force linker to include all object files so that REGISTER() auto-registration
    # in anonymous namespaces is not stripped from the final binary.
    if(UNIX AND NOT APPLE)
        target_link_options(opennn INTERFACE
            "LINKER:--whole-archive" "$<TARGET_FILE:opennn>" "LINKER:--no-whole-archive")
    elseif(APPLE)
        target_link_options(opennn INTERFACE "LINKER:-force_load,$<TARGET_FILE:opennn>")
    elseif(MSVC)
        target_link_options(opennn INTERFACE "/WHOLEARCHIVE:$<TARGET_FILE:opennn>")
    endif()

endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(opennn PUBLIC stdc++fs)
endif()

target_include_directories(opennn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(APPLE)
    target_compile_options(opennn PRIVATE -Xpreprocessor -fopenmp)
elseif(OpenMP_FOUND)
    target_compile_options(opennn PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(opennn PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(opennn PRIVATE -Wno-interference-size)
endif()

if(OPENNN_WITH_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "*.cu" "*.cuh")
    target_sources(opennn PRIVATE ${CUDA_SOURCES})
    target_compile_definitions(opennn PUBLIC WITH_CUDA)
    target_link_libraries(opennn PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_options(opennn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(HAVE_CUDNN)
        target_link_libraries(opennn PUBLIC ${CUDNN_LIBRARY})
        target_include_directories(opennn PUBLIC ${CUDNN_INCLUDE_DIR})
    endif()
endif()

target_compile_options(opennn PRIVATE
    $<$<CONFIG:DEBUG>:-O0 -g>
    $<$<NOT:$<CONFIG:DEBUG>>:-O2>
)

include(GNUInstallDirs)

install(TARGETS opennn
    EXPORT OpenNNTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opennn
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT OpenNNTargets
    NAMESPACE OpenNN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenNN
)
