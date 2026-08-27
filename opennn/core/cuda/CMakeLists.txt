# FlashAttention-2 ships CUDA kernels wrapped in a PyTorch extension. OpenNN
# builds only the kernels and supplies the few ATen/c10 declarations they use.
# The integration is optional because each specialization takes minutes to build.

set(_supported_architectures 80 86 89 90)
set(_requested_architectures "")

foreach(_architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_architecture STREQUAL "native")
        list(APPEND _requested_architectures ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    elseif(_architecture STREQUAL "all")
        list(APPEND _requested_architectures ${CMAKE_CUDA_ARCHITECTURES_ALL})
    elseif(_architecture STREQUAL "all-major")
        list(APPEND _requested_architectures ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
    else()
        list(APPEND _requested_architectures ${_architecture})
    endif()
endforeach()

set(_architectures "")
foreach(_architecture IN LISTS _requested_architectures)
    string(REGEX REPLACE "[^0-9]" "" _number "${_architecture}")
    if(_number IN_LIST _supported_architectures)
        list(APPEND _architectures ${_number})
    endif()
endforeach()
list(REMOVE_DUPLICATES _architectures)

if(NOT _architectures)
    message(STATUS
        "FlashAttention-2: requested CUDA architectures "
        "(${CMAKE_CUDA_ARCHITECTURES}) are unsupported; using cuDNN")
    return()
endif()

if(OpenNN_FLASH_ATTENTION_SOURCE_DIR)
    set(_source_dir "${OpenNN_FLASH_ATTENTION_SOURCE_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        flash_attention
        GIT_REPOSITORY https://github.com/Dao-AILab/flash-attention.git
        GIT_TAG v2.8.3
        GIT_SHALLOW TRUE
        GIT_SUBMODULES csrc/cutlass
        SOURCE_SUBDIR csrc)
    FetchContent_MakeAvailable(flash_attention)
    set(_source_dir "${flash_attention_SOURCE_DIR}")
endif()

set(_kernel_dir "${_source_dir}/csrc/flash_attn/src")
if(NOT EXISTS "${_kernel_dir}/flash_fwd_hdim32_bf16_sm80.cu")
    message(FATAL_ERROR
        "FlashAttention-2 sources were not found at ${_kernel_dir}. Set "
        "OpenNN_FLASH_ATTENTION_SOURCE_DIR or leave it empty to fetch them.")
endif()

set(_sources "")
foreach(_head_dim IN LISTS OpenNN_FLASH_ATTENTION_HEAD_DIMS)
    foreach(_direction fwd bwd)
        foreach(_mask "" "_causal")
            list(APPEND _sources
                "${_kernel_dir}/flash_${_direction}_hdim${_head_dim}_bf16${_mask}_sm80.cu")
        endforeach()
    endforeach()
endforeach()

foreach(_source IN LISTS _sources)
    if(NOT EXISTS "${_source}")
        message(FATAL_ERROR
            "FlashAttention-2 kernel not found: ${_source}. Check "
            "OpenNN_FLASH_ATTENTION_HEAD_DIMS.")
    endif()
endforeach()

add_library(opennn_flash_attention STATIC ${_sources})

set_target_properties(opennn_flash_attention PROPERTIES
    CUDA_ARCHITECTURES "${_architectures}"
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)

target_include_directories(opennn_flash_attention SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/flash_attention_shim>
    $<BUILD_INTERFACE:${_kernel_dir}>
    $<BUILD_INTERFACE:${_source_dir}/csrc/cutlass/include>)

target_compile_definitions(opennn_flash_attention PRIVATE
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_ALIBI
    FLASHATTENTION_DISABLE_LOCAL
    FLASHATTENTION_DISABLE_SOFTCAP)

foreach(_head_dim IN LISTS OpenNN_FLASH_ATTENTION_HEAD_DIMS)
    target_compile_definitions(opennn_flash_attention PUBLIC
        $<BUILD_INTERFACE:OPENNN_FLASH_ATTENTION_HEAD_DIM_${_head_dim}>)
endforeach()

foreach(_architecture IN LISTS _architectures)
    target_compile_definitions(opennn_flash_attention PUBLIC
        $<BUILD_INTERFACE:OPENNN_FLASH_ATTENTION_SM_${_architecture}>)
endforeach()

target_compile_options(opennn_flash_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

target_link_libraries(opennn_flash_attention PRIVATE CUDA::cudart)

message(STATUS
    "FlashAttention-2: head dimensions ${OpenNN_FLASH_ATTENTION_HEAD_DIMS}; "
    "architectures ${_architectures}")
