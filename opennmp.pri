# OpenMP library

win32-msvc {
    QMAKE_CXXFLAGS += /openmp
    QMAKE_LFLAGS += /openmp
}

win32-g++|unix:g++ {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS += -fopenmp
}

clang {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS += -fopenmp
}

macx {
    INCLUDEPATH += /usr/local/opt/libomp/include
    LIBS += -L/usr/local/opt/libomp/lib -lomp
}
