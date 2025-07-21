#   OpenNN: Open Neural Networks Library
#   www.opennn.net                                                                                
#                                                                                                 
#   O P E N N N   E X A M P L E S                                                                 
#                                                                                                 
#   Artificial Intelligence Techniques SL (Artelnics)                                             
#   artelnics@artelnics.com                                                                       

TEMPLATE = subdirs

CONFIG += ordered

# SUBDIRS += airfoil_self_noise
SUBDIRS += forecasting
SUBDIRS += amazon_reviews
# SUBDIRS += breast_cancer
# SUBDIRS += iris_plant
# SUBDIRS += mnist
# SUBDIRS += translation

win32:{
#QMAKE_CXXFLAGS+= -arch:AVX
#QMAKE_CFLAGS+= -arch:AVX
}

# OpenMP library

include(../opennmp.pri)
