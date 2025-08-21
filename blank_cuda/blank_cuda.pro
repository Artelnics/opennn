#   OpenNN: Open Neural Networks Library
#   www.artelnics.com/opennn
#
#   B L A N K _ C U D A   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT += core
TARGET = blank_cuda
TEMPLATE = app

CONFIG += console c++17
CONFIG -= app_bundle

SOURCES += main.cpp

include(../opennmp.pri)

INCLUDEPATH += $$PWD/../opennn

LIBS += -L$$OUT_PWD/../opennn -lopennn

PRE_TARGETDEPS += $$OUT_PWD/../opennn/$$TARGET.a
win32-msvc*: PRE_TARGETDEPS += $$OUT_PWD/../opennn/$$TARGET.lib
