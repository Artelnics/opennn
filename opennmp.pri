win32-msvc* {
    message("OpenMP: Enabled for MSVC")
    QMAKE_CXXFLAGS += /openmp
}

macx {
    message("OpenMP: Configuring for macOS (Clang)")

    OMP_PREFIX = $$system(/opt/homebrew/bin/brew --prefix libomp)

    !isEmpty(OMP_PREFIX) {
        INCLUDEPATH += $$OMP_PREFIX/include
        LIBS        += -L$$OMP_PREFIX/lib -lomp

        QMAKE_CXXFLAGS += -Xpreprocessor -fopenmp
        QMAKE_LFLAGS   += -L$$OMP_PREFIX/lib
    }
}

unix:g++|win32-g++ {
    message("OpenMP: Enabled for GCC (Linux/MinGW)")
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS   += -fopenmp
}
