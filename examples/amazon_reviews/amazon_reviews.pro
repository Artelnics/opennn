###################################################################################################
#                                                                                                 #
#   OpenNN: Open Neural Networks Library                                                          #
#   www.opennn.net                                                                                #
#                                                                                                 #
#   A M A Z O N   R E V I E W S   P R O J E C T                                                    #
#                                                                                                 #
#   Artificial Intelligence Techniques SL (Artelnics)                                             #
#   artelnics@artelnics.com                                                                       #
#                                                                                                 #
###################################################################################################

TEMPLATE = app
CONFIG += console
CONFIG += c++17

mac {
    CONFIG -= app_bundle
}

TARGET = amazon_reviews

DESTDIR = "$$PWD/bin"

SOURCES = main.cpp

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

win32-msvc* {
    CONFIG(debug, debug|release) {
        QMAKE_CXXFLAGS -= /MDd
        QMAKE_CXXFLAGS += /MTd
        QMAKE_CFLAGS -= /MDd
        QMAKE_CFLAGS += /MTd
    } else {
        QMAKE_CXXFLAGS -= /MD
        QMAKE_CXXFLAGS += /MT
        QMAKE_CFLAGS -= /MD
        QMAKE_CFLAGS += /MT
    }
}

win32-msvc* {
    QMAKE_CXXFLAGS += /std:c++17 /openmp
} else {
    QMAKE_CXXFLAGS += -std=c++17 -fopenmp
    QMAKE_LFLAGS += -fopenmp
}

win32-g++ {
    QMAKE_LFLAGS += -static-libgcc
    QMAKE_LFLAGS += -static-libstdc++
    QMAKE_LFLAGS += -static
}

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
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
macx {
    INCLUDEPATH += /usr/local/opt/libomp/include
    LIBS += /usr/local/opt/libomp/lib/libomp.dylib
}
