#   OpenNN: Open Neural Networks Library
#   www.opennn.net                                                                                
#                                                                                                 
#   O P E N N N   E X A M P L E S                                                                 
#                                                                                                 
#   Artificial Intelligence Techniques SL (Artelnics)                                             
#   artelnics@artelnics.com                                                                       

TEMPLATE = subdirs

CONFIG += ordered

SUBDIRS += airfoil_self_noise
SUBDIRS += forecasting
SUBDIRS += amazon_reviews
SUBDIRS += emotion_analysis
SUBDIRS += breast_cancer
SUBDIRS += iris_plant
SUBDIRS += mnist
SUBDIRS += translation

if($$CUDA_ENABLED)
{
    SUBDIRS += melanoma_cancer
}

for(subdir, SUBDIRS) {
    $${subdir}.depends += opennn
}

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}
