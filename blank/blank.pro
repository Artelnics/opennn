#   OpenNN: Open Neural Networks Library
#   www.artelnics.com/opennn
#
#   B L A N K   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

#QT = core
#QT += core widgets

QT += \
    core \
    widgets

TARGET = blank

TEMPLATE = app

CONFIG += console
CONFIG += c++17

mac{
    CONFIG-=app_bundle
}


DESTDIR = "$$PWD/bin"

SOURCES = main.cpp

#win32-g++{
#QMAKE_LFLAGS += -static-libgcc
#QMAKE_LFLAGS += -static-libstdc++
#QMAKE_LFLAGS += -static
#}

#win32:!win32-g++{
##QMAKE_CXXFLAGS+= -arch:AVX
##QMAKE_CFLAGS+= -arch:AVX
#}

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

# OpenNN library

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../opennn/ -lopennn

INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn


win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/opennn.lib
else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/opennn.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../opennn/libopennn.a

# OpenMP library

#win32:{
#QMAKE_CXXFLAGS += -openmp
#QMAKE_LFLAGS  += -openmp
#}

#unix:!macx{
#QMAKE_CXXFLAGS+= -fopenmp
#QMAKE_LFLAGS += -fopenmp

#QMAKE_CXXFLAGS+= -std=c++17
#QMAKE_LFLAGS += -std=c++17
#}
# OpenMP library

include(../opennmp.pri)
