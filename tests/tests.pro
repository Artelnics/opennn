#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   T E S T S
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

TEMPLATE = app
CONFIG += console c++17

TARGET = run_tests
DESTDIR = "$$PWD/bin"

# Google Test
GTEST_DIR = ../googletest
SOURCES += $$GTEST_DIR/src/gtest-all.cc
INCLUDEPATH += $$GTEST_DIR/include $$GTEST_DIR

HEADERS += $$files($$PWD/*.h)
SOURCES += $$files($$PWD/*.cpp)

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}

win32-g++ {
    LIBS += -lwinpthread
}

win32-msvc* {
    QMAKE_CXXFLAGS += /std:c++17 /EHsc /bigobj
}

# OpenN
INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn

win32-msvc {
    CONFIG(debug, debug|release) {
        OPENNN_LIB_PATH = $$OUT_PWD/../opennn/debug
        PRE_TARGETDEPS += $$OPENNN_LIB_PATH/opennn.lib
    } else {
        OPENNN_LIB_PATH = $$OUT_PWD/../opennn/release
        PRE_TARGETDEPS += $$OPENNN_LIB_PATH/opennn.lib
    }
} else:unix|win32-g++ {
    CONFIG(debug, debug|release) {
        OPENNN_LIB_PATH = $$OUT_PWD/../opennn/debug
        PRE_TARGETDEPS += $$OPENNN_LIB_PATH/libopennn.a
    } else {
        OPENNN_LIB_PATH = $$OUT_PWD/../opennn/release
        PRE_TARGETDEPS += $$OPENNN_LIB_PATH/libopennn.a
    }
}
LIBS += -L$$OPENNN_LIB_PATH -lopennn

# CUDA
include(../cuda.pri)

# OpenMP
include(../opennmp.pri)
