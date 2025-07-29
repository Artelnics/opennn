#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   Q T   C R E A T O R   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT = # Do not use qt

TARGET = opennn
#DESTDIR = "$$PWD/bin"

TEMPLATE = lib

CONFIG += staticlib
CONFIG += precompile_header
CONFIG += jumbo_build

PRECOMPILED_HEADER = pch.h
DEFINES += __Cpp17__

#QMAKE_CXXFLAGS += /MP

# OpenMP library

win32-msvc* {
    QMAKE_CXXFLAGS += /std:c++17 /openmp /EHsc
    QMAKE_CXXFLAGS += /bigobj
    DEFINES += "EIGEN_THREAD_LOCAL=thread_local"
}
else:macx {
    QMAKE_CXXFLAGS += -std=c++17
    INCLUDEPATH += /usr/local/opt/libomp/include
    LIBS += -L/usr/local/opt/libomp/lib -lomp
    TEMPLATE_IS_APP = $$find(TEMPLATE, "app")
    !isEmpty(TEMPLATE_IS_APP): LIBS += -lpthread
}
else {
    QMAKE_CXXFLAGS += -std=c++17 -fopenmp
    QMAKE_LFLAGS   += -fopenmp
    TEMPLATE_IS_APP = $$find(TEMPLATE, "app")
    !isEmpty(TEMPLATE_IS_APP): LIBS += -lpthread -lgomp
}

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

#win32:!win32-g++{
##QMAKE_CXXFLAGS+= -arch:AVX
##QMAKE_CFLAGS+= -arch:AVX
#}

#macx{
#INCLUDEPATH += /usr/local/opt/libiomp/include/libiomp
#}

# Eigen library

INCLUDEPATH += ../eigen

# Source files

HEADERS += $$files($$PWD/*.h)
SOURCES += $$files($$PWD/*.cpp)

SOURCES -= $$PWD/addition_layer_4d.cpp
HEADERS -= $$PWD/addition_layer_4d.h
