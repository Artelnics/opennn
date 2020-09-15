#   OpenNN: Open Neural Networks Library
#   www.opennn.net                                                                                
#                                                                                                 
#   O P E N N N   E X A M P L E S                                                                 
#                                                                                                 
#   Artificial Intelligence Techniques SL (Artelnics)                                             
#   artelnics@artelnics.com                                                                       

TEMPLATE = subdirs

CONFIG += ordered

SUBDIRS += rosenbrock
SUBDIRS += simple_approximation
SUBDIRS += simple_classification
#SUBDIRS += airfoil_self_noise
#SUBDIRS += airline_passengers
#SUBDIRS += breast_cancer
#SUBDIRS += iris_plant
#SUBDIRS += logical_operations
#SUBDIRS += pima_indians_diabetes
#SUBDIRS += urinary_inflammations_diagnosis
#SUBDIRS += yacht_hydrodynamics_design
#SUBDIRS += yacht_hydrodynamics_production
#SUBDIRS += leukemia
#SUBDIRS += mnist

# OpenMP library

win32:!win32-g++{
QMAKE_CXXFLAGS += -std=c++11 -fopenmp -pthread #-lgomp -openmp
QMAKE_LFLAGS += -fopenmp -pthread #-lgomp -openmp
LIBS += -fopenmp -pthread #-lgomp
}else:!macx{QMAKE_CXXFLAGS+= -fopenmp -lgomp -std=c++11
QMAKE_LFLAGS += -fopenmp -lgomp
LIBS += -fopenmp -pthread -lgomp
}else: macx{
INCLUDEPATH += /usr/local/opt/libomp/include
LIBS += /usr/local/opt/libomp/lib/libomp.dylib}
