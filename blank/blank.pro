#   OpenNN: Open Neural Networks Library
#   www.artelnics.com/opennn
#
#   B L A N K   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT += core widgets

TARGET = blank
TEMPLATE = app
CONFIG += console c++17

mac { CONFIG -= app_bundle }

DESTDIR = "$$PWD/bin"

SOURCES = main.cpp

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}

CUDA_PATH = $$(CUDA_PATH)
isEmpty(CUDA_PATH): CUDA_PATH = $$(CUDA_HOME)
win32: isEmpty(CUDA_PATH) {
    CUDA_BASE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    CUDA_VERSIONS_FOUND = $$files($$CUDA_BASE_DIR/v*, true)
    !isEmpty(CUDA_VERSIONS_FOUND): CUDA_PATH = $$last(CUDA_VERSIONS_FOUND)
}

if(!isEmpty(CUDA_PATH)) {
    CUDA_PATH = $$clean_path($$CUDA_PATH)
    CUDA_INCLUDE_PATH = $$CUDA_PATH/include
    CUDA_LIB_DIR = $$CUDA_PATH/lib/x64

    INCLUDEPATH += $$CUDA_INCLUDE_PATH
    DEPENDPATH += $$CUDA_INCLUDE_PATH
    LIBS += -L$$CUDA_LIB_DIR -lcudart -lcublas

    exists($$CUDA_INCLUDE_PATH/cudnn.h) {
        LIBS += -lcudnn
    }
}

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../opennn/ -lopennn

INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/libopennn.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/libopennn.a
else:win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/opennn.lib
else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/opennn.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../opennn/libopennn.a

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp
include(../opennmp.pri)
