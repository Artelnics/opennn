# FlashAttention-2's kernels, compiled without PyTorch.
#
# FA2 ships as CUDA sources plus a PyTorch extension; only the sources are used
# here. They reach for a handful of ATen/c10 declarations, which
# flash_attention_shim/ supplies - two launch-check macros and a philox state
# that no dropout-free call ever unpacks.
#
# The kernels are Ampere through Hopper (sm_80 to sm_90); Blackwell has no FA2
# kernel in this release, so the target is built for whichever of the requested
# architectures FA2 covers, and skipped entirely when none of them is covered.
#
# Compiling one kernel takes minutes, and there is one per head dimension per
# mask per direction, which is why OpenNN_FLASH_ATTENTION_HEAD_DIMS exists: a
# build that only ever sees d_model/heads == 64 needs one quarter of the list.

set(OPENNN_FLASH_ATTENTION_ARCHITECTURES 80 86 89 90)

# Where this file is, remembered here: inside the function below,
# CMAKE_CURRENT_LIST_DIR is whichever file called it.
set(OPENNN_FLASH_ATTENTION_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(opennn_add_flash_attention_kernels out_target)

    # What the build was asked for, with the keywords resolved to numbers (the
    # default is "native") and the decorations dropped ("86-real" is 86).
    set(requested "")
    foreach(architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(architecture STREQUAL "native")
            list(APPEND requested ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
        elseif(architecture STREQUAL "all")
            list(APPEND requested ${CMAKE_CUDA_ARCHITECTURES_ALL})
        elseif(architecture STREQUAL "all-major")
            list(APPEND requested ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
        else()
            list(APPEND requested ${architecture})
        endif()
    endforeach()

    set(architectures "")
    foreach(architecture IN LISTS requested)
        string(REGEX REPLACE "[^0-9]" "" number "${architecture}")
        if(number IN_LIST OPENNN_FLASH_ATTENTION_ARCHITECTURES)
            list(APPEND architectures ${number})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES architectures)

    if(NOT architectures)
        message(STATUS
            "FlashAttention-2: no requested CUDA architecture is one FA2 covers "
            "(${CMAKE_CUDA_ARCHITECTURES} against ${OPENNN_FLASH_ATTENTION_ARCHITECTURES}); "
            "attention keeps the cuDNN path")
        set(${out_target} "" PARENT_SCOPE)
        return()
    endif()

    if(OpenNN_FLASH_ATTENTION_SOURCE_DIR)
        set(source_dir "${OpenNN_FLASH_ATTENTION_SOURCE_DIR}")
    else()
        include(FetchContent)
        FetchContent_Declare(
            flash_attention
            GIT_REPOSITORY https://github.com/Dao-AILab/flash-attention.git
            GIT_TAG        v2.8.3
            GIT_SHALLOW    TRUE
            GIT_SUBMODULES csrc/cutlass
            # csrc has no CMakeLists.txt, which is how the population stays a
            # download: FA2's own build is a setup.py, not a subproject.
            SOURCE_SUBDIR  csrc
        )
        FetchContent_MakeAvailable(flash_attention)
        set(source_dir "${flash_attention_SOURCE_DIR}")
    endif()

    set(kernel_dir "${source_dir}/csrc/flash_attn/src")
    if(NOT EXISTS "${kernel_dir}/flash_fwd_hdim32_bf16_sm80.cu")
        message(FATAL_ERROR
            "FlashAttention-2 sources are not at ${kernel_dir}. Point "
            "OpenNN_FLASH_ATTENTION_SOURCE_DIR at a flash-attention checkout, "
            "or leave it unset to have one fetched.")
    endif()

    set(sources "")
    foreach(head_dim IN LISTS OpenNN_FLASH_ATTENTION_HEAD_DIMS)
        foreach(direction fwd bwd)
            foreach(mask "" "_causal")
                list(APPEND sources
                     "${kernel_dir}/flash_${direction}_hdim${head_dim}_bf16${mask}_sm80.cu")
            endforeach()
        endforeach()
    endforeach()

    foreach(source IN LISTS sources)
        if(NOT EXISTS "${source}")
            message(FATAL_ERROR "FlashAttention-2: no kernel source at ${source} "
                                "(OpenNN_FLASH_ATTENTION_HEAD_DIMS names a head "
                                "dimension FA2 does not ship).")
        endif()
    endforeach()

    add_library(opennn_flash_attention STATIC ${sources})

    set_target_properties(opennn_flash_attention PROPERTIES
        CUDA_ARCHITECTURES "${architectures}"
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)

    # Public because the caller of these kernels (core/cuda/flash_attention.cu)
    # fills FA2's own parameter struct and names its element type, so it needs
    # the same three headers the kernels are built against. Only while building,
    # though: none of this reaches an installed OpenNN, whose headers name no
    # FA2 type and whose consumers have no FA2 checkout to point at.
    target_include_directories(opennn_flash_attention SYSTEM PUBLIC
        $<BUILD_INTERFACE:${OPENNN_FLASH_ATTENTION_MODULE_DIR}/flash_attention_shim>
        $<BUILD_INTERFACE:${kernel_dir}>
        $<BUILD_INTERFACE:${source_dir}/csrc/cutlass/include>)

    # Everything FA2 can do that this integration does not ask for. Each one
    # left in costs compile time in template instantiations that are never
    # launched: the rung refuses dropout, and nothing here attends over a
    # window, an alibi slope, or a softcap.
    target_compile_definitions(opennn_flash_attention PRIVATE
        FLASHATTENTION_DISABLE_DROPOUT
        FLASHATTENTION_DISABLE_ALIBI
        FLASHATTENTION_DISABLE_LOCAL
        FLASHATTENTION_DISABLE_SOFTCAP)

    # What was actually built, for the caller to dispatch on: a head dimension
    # left out of the list has no kernel to link against, and an architecture
    # left out has no kernel image to launch.
    foreach(head_dim IN LISTS OpenNN_FLASH_ATTENTION_HEAD_DIMS)
        target_compile_definitions(opennn_flash_attention PUBLIC
            $<BUILD_INTERFACE:OPENNN_FLASH_ATTENTION_HEAD_DIM_${head_dim}>)
    endforeach()

    foreach(architecture IN LISTS architectures)
        target_compile_definitions(opennn_flash_attention PUBLIC
            $<BUILD_INTERFACE:OPENNN_FLASH_ATTENTION_SM_${architecture}>)
    endforeach()

    target_compile_options(opennn_flash_attention PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

    target_link_libraries(opennn_flash_attention PRIVATE CUDA::cudart)

    message(STATUS
        "FlashAttention-2: kernels for head dims ${OpenNN_FLASH_ATTENTION_HEAD_DIMS} "
        "on sm ${architectures}")

    set(${out_target} opennn_flash_attention PARENT_SCOPE)

endfunction()
