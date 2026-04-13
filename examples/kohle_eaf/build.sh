#!/bin/bash
# Build script for kohle_eaf example
export PATH="/c/Qt/Tools/mingw1310_64/bin:$PATH"

OPENNN_DIR="../../opennn"
EIGEN_DIR="../../eigen"
LIB_DIR="../../build/Desktop_Qt_6_9_3_MinGW_64_bit-Release/opennn/release"
QT_DIR="C:/Qt/6.9.3/mingw_64"

echo "Compiling kohle_eaf..."

g++ -fopenmp -O2 -std=gnu++2a -Wall -fexceptions -mthreads \
    -DUNICODE -D_UNICODE -DNDEBUG -DQT_NO_DEBUG -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB \
    -I${OPENNN_DIR} -I${EIGEN_DIR} \
    -I${QT_DIR}/include -I${QT_DIR}/include/QtCore \
    -o kohle_eaf.exe main.cpp \
    -L${LIB_DIR} -lopennn \
    -fopenmp -static-libgcc -static-libstdc++ -static \
    ${QT_DIR}/lib/libQt6Widgets.a ${QT_DIR}/lib/libQt6Gui.a ${QT_DIR}/lib/libQt6Core.a

echo "Build result: $?"
