###################################################################################################
#                                                                                                 #
#   OpenNN: Open Neural Networks Library                                                          #
#   www.artelnics.com/opennn                                                                      #
#                                                                                                 #
#   Y A C H T   H Y D R O D Y N A M I C S   P R O D U C T I O N   P R O J E C T                   #
#                                                                                                 #
#   Artificial Intelligence Techniques SL (Artelnics)                                             #
#   artelnics@artelnics.com                                                                       #
#                                                                                                 #
###################################################################################################

TEMPLATE = app
CONFIG += console
CONFIG += c++11

mac{
    CONFIG-=app_bundle
}

TARGET = yacht_hydrodynamics_production

DESTDIR = "$$PWD/bin"

SOURCES = main.cpp

win32-g++{
QMAKE_LFLAGS += -static-libgcc
QMAKE_LFLAGS += -static-libstdc++
QMAKE_LFLAGS += -static

QMAKE_CXXFLAGS += -std=c++11 -fopenmp -pthread -lgomp
QMAKE_LFLAGS += -fopenmp -pthread -lgomp
LIBS += -fopenmp -pthread -lgomp
}

# OpenNN library

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../../opennn/ -lopennn

INCLUDEPATH += $$PWD/../../opennn
DEPENDPATH += $$PWD/../../opennn

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../opennn/release/libopennn.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../opennn/debug/libopennn.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../opennn/release/opennn.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../opennn/debug/opennn.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../../opennn/libopennn.a

# OpenMP library
!win32{
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp
}

mac{
INCLUDEPATH += /usr/local/Cellar/libiomp/20150701/include/libiomp
LIBS += -L/usr/local/Cellar/libiomp/20150701/lib -liomp5
}

# MPI include
#include(../../mpi.pri)

# CUDA include
#include(../../cuda.pri)
