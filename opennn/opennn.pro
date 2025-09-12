#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   Q T   C R E A T O R   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT = # Do not use qt

TARGET = opennn
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

# Eigen library

INCLUDEPATH += ../eigen

# Source files

PRECOMPILED_HEADER = pch.h
HEADERS += $$files($$PWD/*.h)
SOURCES += $$files($$PWD/*.cpp)

# CUDA_SOURCES += kernel.cu

# include(../cuda.pri)

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}
