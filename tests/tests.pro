#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   T E S T S
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

# Project Configuration
TEMPLATE = app
CONFIG += console c++17

# Project Name
TARGET = run_tests
DESTDIR = "$$PWD/bin"

DEFINES += EIGEN_DONT_PARALLELIZE

# Google Test configuration
GTEST_DIR = ../googletest
SOURCES += $$GTEST_DIR/src/gtest-all.cc
INCLUDEPATH += $$GTEST_DIR/include $$GTEST_DIR

HEADERS += $$files($$PWD/*.h)
SOURCES += $$files($$PWD/*.cpp)

# OpenNN library linking
win32 {
    CONFIG(debug, debug|release) {
        LIBS += -L$$OUT_PWD/../opennn/debug -lopennn
        PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/libopennn.a
    } else {
        LIBS += -L$$OUT_PWD/../opennn/release -lopennn
        PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/libopennn.a
    }
} else:unix {
    LIBS += -L$$OUT_PWD/../opennn/ -lopennn
    PRE_TARGETDEPS += $$OUT_PWD/../opennn/libopennn.a
}

INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn

# OpenMP
include(../opennmp.pri)
win32-g++|linux-g++ {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS += -fopenmp
} else:win32-msvc* {
    QMAKE_CXXFLAGS += /openmp
} else:macx {
    LIBS += -lomp
}
