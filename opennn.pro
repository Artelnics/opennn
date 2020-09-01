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
