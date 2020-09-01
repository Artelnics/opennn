#   OpenNN: Open Neural Networks Library
#   www.artelnics.com/opennn
#
#   B L A N K   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT = core# Do not use qt

TARGET = blank

TEMPLATE = app

CONFIG += console
CONFIG += c++11

mac{
    CONFIG-=app_bundle
}


DESTDIR = "$$PWD/bin"

SOURCES += main.cpp

win32-g++{
QMAKE_LFLAGS += -static-libgcc
QMAKE_LFLAGS += -static-libstdc++
QMAKE_LFLAGS += -static
}

# OpenNN library

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../opennn/ -lopennn

INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/libopennn.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/libopennn.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/opennn.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/opennn.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../opennn/libopennn.a

# OpenMP library

win32:!win32-g++{
QMAKE_CXXFLAGS += -openmp
QMAKE_LFLAGS  += -openmp
}

unix:!macx{
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS += -fopenmp

QMAKE_CXXFLAGS+= -std=c++11
QMAKE_LFLAGS += -std=c++11
}

# OpenMP Windows Visual C++

win32:!win32-g++{
QMAKE_CXXFLAGS += -std=c++11 -fopenmp -pthread #-lgomp
QMAKE_LFLAGS += -fopenmp -pthread #-lgomp
LIBS += -fopenmp -pthread #-lgomp
}else:!macx{
QMAKE_CXXFLAGS+= -fopenmp #-lgomp
QMAKE_LFLAGS += -fopenmp #-lgomp
LIBS += -openmp -pthread #-lgomp
}else: macx{
INCLUDEPATH += /usr/local/opt/libomp/include
LIBS += /usr/local/opt/libomp/lib/libomp.dylib
}
