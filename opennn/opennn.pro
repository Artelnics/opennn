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

# CUDA

win32-msvc* {
    CUDA_SOURCES += kernel.cu
}

include(../cuda.pri)

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}
