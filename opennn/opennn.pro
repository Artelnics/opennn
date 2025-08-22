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

DEFINES += __Cpp17__

win32-msvc* {
    QMAKE_CXXFLAGS += /std:c++17 /openmp /EHsc
    QMAKE_CXXFLAGS += /bigobj
    #DEFINES += "EIGEN_THREAD_LOCAL=thread_local"
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

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}
CONFIG(release, debug|release): DEFINES += NDEBUG

CUDA_PATH = $$(CUDA_PATH)
isEmpty(CUDA_PATH): CUDA_PATH = $$(CUDA_HOME)
win32: isEmpty(CUDA_PATH) {
    CUDA_BASE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    CUDA_VERSIONS_FOUND = $$files($$CUDA_BASE_DIR/v*, true)
    !isEmpty(CUDA_VERSIONS_FOUND): CUDA_PATH = $$last(CUDA_VERSIONS_FOUND)
}

if(!isEmpty(CUDA_PATH)) {
    CUDA_PATH = $$clean_path($$CUDA_PATH)
    NVCC_EXECUTABLE_TEST = $$CUDA_PATH/bin/nvcc.exe
    CUDA_INCLUDE_PATH_TEST = $$CUDA_PATH/include
    CUDA_LIB_DIR_TEST = $$CUDA_PATH/lib/x64

    if(exists($$NVCC_EXECUTABLE_TEST)) {
        if(exists($$CUDA_INCLUDE_PATH_TEST)) {
            if(exists($$CUDA_LIB_DIR_TEST)) {

                message("[opennn.pro] Valid CUDA , configuring...")

                DEFINES += WITH_CUDA
                CUDA_INCLUDE_PATH = $$CUDA_INCLUDE_PATH_TEST
                CUDA_LIB_DIR = $$CUDA_LIB_DIR_TEST
                NVCC_EXECUTABLE = $$NVCC_EXECUTABLE_TEST

                INCLUDEPATH += $$CUDA_INCLUDE_PATH
                DEPENDPATH += $$CUDA_INCLUDE_PATH
                LIBS += -L$$CUDA_LIB_DIR -lcudart -lcublas

                exists($$CUDA_INCLUDE_PATH/cudnn.h) {
                    message("    -> adding cuDNN .")
                    DEFINES += HAVE_CUDNN
                    LIBS += -lcudnn
                }

                CUDA_COMPILER = $$NVCC_EXECUTABLE
                CUDA_ARCH = -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75
                CONFIG(debug, debug|release): CUDA_FLAGS = -g -G
                else: CUDA_FLAGS = --expt-relaxed-constexpr -O2

                cuda.input = CUDA_SOURCES
                CONFIG(debug, debug|release): cuda.output = $$OUT_PWD/debug/${QMAKE_FILE_BASE}.obj
                else: cuda.output = $$OUT_PWD/release/${QMAKE_FILE_BASE}.obj

                CUDA_HOST_CXXFLAGS = /std:c++17 /openmp /EHsc /bigobj

                cuda.dependency_type = TYPE_C
                cuda.variable_out = OBJECTS
                QMAKE_EXTRA_COMPILERS += cuda

                CUDA_SOURCES += $$files($$PWD/*.cu)
            }
        }
    }
}

# Eigen library

INCLUDEPATH += ../eigen

# Source files

PRECOMPILED_HEADER = pch.h
HEADERS += $$files($$PWD/*.h)
SOURCES += $$files($$PWD/*.cpp)
