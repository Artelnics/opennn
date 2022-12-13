# OpenMP library

win32:{
QMAKE_CXXFLAGS += -d2ReducedOptimizeHugeFunctions
QMAKE_CXXFLAGS += -std=c++17 -fopenmp -pthread #-lgomp -openmp
QMAKE_LFLAGS += -fopenmp -pthread #-lgomp -openmp
LIBS += -fopenmp -pthread #-lgomp
#QMAKE_CXXFLAGS+= -arch:AVX
#QMAKE_CFLAGS+= -arch:AVX
QMAKE_CXXFLAGS += -openmp
QMAKE_LFLAGS  += -openmp
QMAKE_CXXFLAGS += -MP
}

unix:!macx{QMAKE_CXXFLAGS+= -fopenmp -lgomp -std=c++17
QMAKE_LFLAGS += -fopenmp -lgomp
LIBS += -fopenmp -pthread -lgomp
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS += -fopenmp
QMAKE_CXXFLAGS+= -std=c++17
QMAKE_LFLAGS += -std=c++17
}
unix:macx{
INCLUDEPATH += /usr/local/opt/libomp/include
LIBS += /usr/local/opt/libomp/lib/libomp.dylib}
