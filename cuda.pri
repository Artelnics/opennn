# cuda.pri

isEmpty(CUDA_PATH_DETECTED) {
    CUDA_PATH_DETECTED = true

    message("--- Running CUDA detection from cuda.pri ---")

    CUDA_PATH = $$(CUDA_PATH)
    isEmpty(CUDA_PATH): CUDA_PATH = $$(CUDA_HOME)

    win32: isEmpty(CUDA_PATH) {
        CUDA_BASE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
        CUDA_VERSIONS_FOUND = $$files($$CUDA_BASE_DIR/v*, true)
        !isEmpty(CUDA_VERSIONS_FOUND): CUDA_PATH = $$last(CUDA_VERSIONS_FOUND)
    }
}


if(!isEmpty(CUDA_PATH)) {
    CUDA_PATH = $$clean_path($$CUDA_PATH)

    win32: NVCC_EXECUTABLE_TEST = $$CUDA_PATH/bin/nvcc.exe
    else:  NVCC_EXECUTABLE_TEST = $$CUDA_PATH/bin/nvcc

    CUDA_INCLUDE_PATH_TEST = $$CUDA_PATH/include

    win32: CUDA_LIB_DIR_TEST = $$CUDA_PATH/lib/x64
    else:  CUDA_LIB_DIR_TEST = $$CUDA_PATH/lib64

    if(exists($$NVCC_EXECUTABLE_TEST):exists($$CUDA_INCLUDE_PATH_TEST):exists($$CUDA_LIB_DIR_TEST)) {

        message("    -> CUDA found at: $$CUDA_PATH.")

        CUDA_ENABLED = true
        NVCC_EXECUTABLE = $$NVCC_EXECUTABLE_TEST
        CUDA_INCLUDE_PATH = $$CUDA_INCLUDE_PATH_TEST
        CUDA_LIB_DIR = $$CUDA_LIB_DIR_TEST

        DEFINES += WITH_CUDA

        INCLUDEPATH += $$CUDA_INCLUDE_PATH
        DEPENDPATH += $$CUDA_INCLUDE_PATH
        LIBS += -L$$CUDA_LIB_DIR -lcudart -lcublas

        exists($$CUDA_INCLUDE_PATH/cudnn.h) {
            message("    -> cuDNN found. Adding to build.")
            DEFINES += HAVE_CUDNN
            LIBS += -lcudnn
        }

        if(!isEmpty(CUDA_SOURCES)) {
            message("    -> Configuring NVCC for CUDA sources (.cu files)...")

            # Flags para el compilador de CUDA (NVCC)
            NVCC_FLAGS = --use_fast_math
            NVCC_FLAGS += --std=c++17
            NVCC_FLAGS += --expt-relaxed-constexpr
            NVCC_FLAGS += -gencode arch=compute_61,code=sm_61
            NVCC_FLAGS += -gencode arch=compute_75,code=sm_75
            NVCC_FLAGS += -gencode arch=compute_86,code=sm_86

            win32: NVCC_FLAGS += -Xcompiler "/MD"
            unix: NVCC_FLAGS += -Xcompiler -fPIC

            debug:   NVCC_FLAGS_DEBUG   = -g -G
            release: NVCC_FLAGS_RELEASE = --ptxas-options=-v

            cuda.commands = $$NVCC_EXECUTABLE -c $$NVCC_FLAGS \
                            $$join(NVCC_FLAGS_RELEASE, " ", "", " ") \
                            $$join(NVCC_FLAGS_DEBUG, " ", "", " ") \
                            ${QMAKE_FILE_IN} -o ${QMAKE_FILE_OUT}

            isEmpty(OBJECTS_DIR) {
                CONFIG(debug, debug|release) {
                    OBJECTS_DIR = $$OUT_PWD/debug
                } else {
                    OBJECTS_DIR = $$OUT_PWD/release
                }
            }

            !exists($$OBJECTS_DIR) {
                mkpath($$OBJECTS_DIR)
            }

            win32: cuda.output = $$shell_path($$OBJECTS_DIR)/${QMAKE_FILE_BASE}.obj
            else:  cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.o

            cuda.input = CUDA_SOURCES
            cuda.variable_out = OBJECTS
            cuda.dependency_type = TYPE_C
            cuda.clean = $$OBJECTS_DIR\\${QMAKE_FILE_BASE}.obj $$OBJECTS_DIR/${QMAKE_FILE_BASE}.o

            QMAKE_EXTRA_COMPILERS += cuda
        }
    } else {
        message("--- CUDA path found, but essential directories/files are missing. CUDA support disabled. ---")
    }
} else {
    message("--- CUDA not found. Building without CUDA support. ---")
}
