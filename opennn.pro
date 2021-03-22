#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

# CONFIGURATION

TEMPLATE = subdirs

CONFIG(release, debug|release) {
DEFINES += NDEBUG
}

SUBDIRS += opennn

SUBDIRS += tests
SUBDIRS += examples
SUBDIRS += blank

CONFIG += ordered

win32:!win32-g++{
#QMAKE_CXXFLAGS+= -arch:AVX
#QMAKE_CFLAGS+= -arch:AVX
}

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
