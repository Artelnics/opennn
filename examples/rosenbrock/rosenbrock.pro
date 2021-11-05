#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   R O S E N B R O C K   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

TEMPLATE = app
CONFIG += console
CONFIG += c++11

mac{
    CONFIG-=app_bundle
}

TARGET = rosenbrock

DESTDIR = "$$PWD/bin"

SOURCES = main.cpp

win32-g++{
QMAKE_LFLAGS += -static-libgcc
QMAKE_LFLAGS += -static-libstdc++
QMAKE_LFLAGS += -static

win32:!win32-g++{
#QMAKE_CXXFLAGS+= -arch:AVX
#QMAKE_CFLAGS+= -arch:AVX
}

#QMAKE_CXXFLAGS += -std=c++11 -fopenmp -pthread -lgomp
#QMAKE_LFLAGS += -fopenmp -pthread -lgomp
#LIBS += -fopenmp -pthread -lgomp
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

INCLUDEPATH += D:/OpenNN/eigen

# OpenMP library

win32:!win32-g++{
QMAKE_CXXFLAGS += -std=c++11 -fopenmp -pthread #-lgomp -openmp
QMAKE_LFLAGS += -fopenmp -pthread #-lgomp -openmp
LIBS += -fopenmp -pthread #-lgomp
}else:!macx{QMAKE_CXXFLAGS+= -fopenmp -lgomp -std=c++11
QMAKE_LFLAGS += -fopenmp -lgomp
LIBS += -fopenmp -pthread -lgomp
}else: macx{
INCLUDEPATH += /usr/local/opt/libomp/include
LIBS += /usr/local/opt/libomp/lib/libomp.dylib}
#DEFINES += __OPENNN_CUDA__

#contains(DEFINES, __OPENNN_CUDA__){
#    include(../../../Artelnics/opennn_cuda/cuda_config.pri)
#    include(../../../Artelnics/opennn_cuda/cuda_path.pri)
#}


#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../openblas/lib/ -llibopenblas
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../openblas/lib/ -llibopenblasd
#else:unix: LIBS += -L$$PWD/../../../openblas/lib/ -llibopenblas

#INCLUDEPATH += $$PWD/../../../openblas/include
#DEPENDPATH += $$PWD/../../../openblas/include

#win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../openblas/lib/liblibopenblas.a
#else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../openblas/lib/liblibopenblasd.a
#else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../openblas/lib/libopenblas.lib
#else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../openblas/lib/libopenblasd.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../../../openblas/lib/liblibopenblas.a
