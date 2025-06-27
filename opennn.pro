#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

# CONFIGURATION

TEMPLATE = subdirs

CONFIG(release, debug|release) {
DEFINES += NDEBUG
}

SUBDIRS += opennn
SUBDIRS += examples
#SUBDIRS += blank
#SUBDIRS += blank_cuda
#SUBDIRS += tests

CONFIG += ordered

# --- CUDA 12.5 ---
MY_CUDA_VER_MAJOR = 12
MY_CUDA_VER_MINOR = 5

MY_CUDA_FULL_VERSION_STR = $$sprintf("v%1.%2", $$MY_CUDA_VER_MAJOR, $$MY_CUDA_VER_MINOR)
MY_CUDA_SHORT_VERSION_STR = $$sprintf("%1.%2", $$MY_CUDA_VER_MAJOR, $$MY_CUDA_VER_MINOR)

win32 {
    CUDA_PATH_ATTEMPT1 = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/$$MY_CUDA_FULL_VERSION_STR"
    CUDA_PATH_ATTEMPT2 = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/$$MY_CUDA_SHORT_VERSION_STR"

    CUDA_PATH = $$CUDA_PATH_ATTEMPT1
    !exists($$CUDA_PATH) {
        CUDA_PATH = $$CUDA_PATH_ATTEMPT2
    }

    CUDA_LIB_DIR = $$CUDA_PATH/lib/x64
    CUDA_BIN_DIR = $$CUDA_PATH/bin
} else:unix {
    CUDA_PATH_ATTEMPT1_UNIX = /usr/local/cuda-$$MY_CUDA_SHORT_VERSION_STR
    CUDA_PATH_ATTEMPT2_UNIX = /usr/local/cuda-$$MY_CUDA_FULL_VERSION_STR
    CUDA_PATH_ATTEMPT3_UNIX = /usr/local/cuda

    CUDA_PATH = $$CUDA_PATH_ATTEMPT1_UNIX
    !exists($$CUDA_PATH) {
        CUDA_PATH = $$CUDA_PATH_ATTEMPT2_UNIX
        !exists($$CUDA_PATH) {
            CUDA_PATH = $$CUDA_PATH_ATTEMPT3_UNIX
        }
    }
    CUDA_LIB_DIR = $$CUDA_PATH/lib
    CUDA_BIN_DIR = $$CUDA_PATH/bin
}

!exists($$CUDA_PATH) {
    warning("CUDA path not found: $$CUDA_PATH. Please check your CUDA installation and CUDA_PATH in the .pro file.")
    warning("Attempted versions based on MY_VARS: $$MY_CUDA_FULL_VERSION_STR and $$MY_CUDA_SHORT_VERSION_STR")
} else {
    message("Using CUDA from: $$CUDA_PATH")
}

INCLUDEPATH += $$CUDA_PATH/include
DEPENDPATH += $$CUDA_PATH/include

LIBS += -L$$CUDA_LIB_DIR -lcudart_static

# OpenMP library
include(../opennn/opennmp.pri)
