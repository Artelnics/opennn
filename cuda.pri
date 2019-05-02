# Path to cuda toolkit install

windows{
CUDA_DIR = C:/"Program Files"/"NVIDIA GPU Computing Toolkit"/CUDA/v10.0
}else:mac{
CUDA_DIR = /Developer/NVIDIA/CUDA-7.5
}else:unix{
CUDA_DIR = /usr/local/cuda-7.5
}

windows{
QMAKE_CXXFLAGS_RELEASE -= -MD
QMAKE_CXXFLAGS_DEBUG -= -MDd

QMAKE_CXXFLAGS_RELEASE += /MT
QMAKE_CXXFLAGS_DEBUG += /MT
QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib
}

windows: LIBS += -L$$CUDA_DIR/lib/x64/ -lcuda

windows: LIBS += -L$$CUDA_DIR/lib/x64/ -lcudart

windows:CONFIG(release, debug|release): LIBS += -L$$CUDA_DIR/lib/x64/ -lcublas
else:windows:CONFIG(debug, debug|release): LIBS += -L$$CUDA_DIR/lib/x64/ -lcublas

windows:CONFIG(release, debug|release): LIBS += -L$$CUDA_DIR/lib/x64/ -lcurand
else:windows:CONFIG(debug, debug|release): LIBS += -L$$CUDA_DIR/lib/x64/ -lcurand

macx: LIBS += -L$$CUDA_DIR/lib/ -lcudart

macx:CONFIG(release, debug|release): LIBS += -L$$CUDA_DIR/lib/ -lcublas
else:macx:CONFIG(debug, debug|release): LIBS += -L$$CUDA_DIR/lib/ -lcublas


unix:!macx: LIBS += -L$$CUDA_DIR/lib64/ -lcudart

INCLUDEPATH += $$CUDA_DIR/lib64
DEPENDPATH += $$CUDA_DIR/lib64

unix:!macx: LIBS += -L$$CUDA_DIR/lib64/ -lcublas

INCLUDEPATH += $$CUDA_DIR/lib64
DEPENDPATH += $$CUDA_DIR/lib64
