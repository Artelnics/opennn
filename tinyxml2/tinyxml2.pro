###################################################################################################
#                                                                                                 #
#   OpenNN: Open Neural Networks Library                                                          #
#   www.opennn.net                                                                      #
#                                                                                                 #
#   T I N Y   X M L   2   Q T   C R E A T O R   P R O J E C T                                     #
#                                                                                                 #
#   Roberto Lopez                                                                                 #
#   Artelnics - Making intelligent use of data                                                    #
#   robertolopez@artelnics.com                                                                    #
#                                                                                                 #
###################################################################################################

QT = # Do not use qt

greaterThan(QT_MAJOR_VERSION, 4): CONFIG += c++11
lessThan(QT_MAJOR_VERSION, 5): QMAKE_CXXFLAGS += -std=c++11

TARGET = tinyxml2

TEMPLATE = lib

CONFIG += staticlib

HEADERS += tinyxml2.h

SOURCES += tinyxml2.cpp

