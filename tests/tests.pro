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

# Enable precompiled headers in Qt
QMAKE_CXXFLAGS += -include pch.h

# OpenNN library
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../opennn/ -lopennn

INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/libopennn.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/libopennn.a
else:win32-msvc*:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/opennn.lib
else:win32-msvc*:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/opennn.lib
else: PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/libopennn.a

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
