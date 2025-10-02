#   OpenNN: Open Neural Networks Library
#   www.opennn.net                                                                                
#                                                                                                 
#   O P E N N N   E X A M P L E S                                                                 
#                                                                                                 
#   Artificial Intelligence Techniques SL (Artelnics)                                             
#   artelnics@artelnics.com                                                                       

# Project Configuration
TEMPLATE = app
CONFIG += console c++17

# Project Name
TARGET = run_tests

TEMPLATE = app

DESTDIR = "$$PWD/bin"

GTEST_DIR = ../googletest

SOURCES += $$GTEST_DIR/src/gtest-all.cc
INCLUDEPATH += $$GTEST_DIR/include
INCLUDEPATH += $$GTEST_DIR

HEADERS += $$files($$PWD/*.h)
SOURCES += $$files($$PWD/*.cpp)

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}

# Enable precompiled headers in Qt
QMAKE_CXXFLAGS += -include pch.h

# OpenNN library
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../opennn/ -lopennn

INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn

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

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/libopennn.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/libopennn.a
else:win32-msvc*:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/opennn.lib
else:win32-msvc*:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/opennn.lib
else: PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/libopennn.a

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp
include(../opennmp.pri)
