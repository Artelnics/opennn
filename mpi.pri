# Path to cuda toolkit install

DEFINES += __OPENNN_MPI__

win32{
MPI_DIR = C:/"Program Files (x86)"/"Microsoft SDKs"/MPI

INCLUDEPATH += $$MPI_DIR/Include
DEPENDPATH += $$MPI_DIR/Include

LIBS += -L$$MPI_DIR/Lib/x64/ -lmsmpi

INCLUDEPATH += $$MPI_DIR/Lib/x64
DEPENDPATH += $$MPI_DIR/Lib/x64

}else:mac{
MPI_DIR =
}else:unix{

# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc

#QMAKE_CFLAGS += $$system(mpicc --showme:compile)
#QMAKE_LFLAGS += $$system(mpicxx --showme:link)
#QMAKE_CXXFLAGS += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
#QMAKE_CXXFLAGS_RELEASE += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK

MPI_DIR = /usr/lib/openmpi

INCLUDEPATH += $$MPI_DIR/include
DEPENDPATH += $$MPI_DIR/include

LIBS += -L$$MPI_DIR/lib/ -lmpi

INCLUDEPATH += $$MPI_DIR/lib
DEPENDPATH += $$MPI_DIR/lib
}
