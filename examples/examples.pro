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
# SUBDIRS += forecasting
SUBDIRS += amazon_reviews
SUBDIRS += emotion_analysis
# SUBDIRS += breast_cancer
SUBDIRS += iris_plant
# SUBDIRS += mnist
# SUBDIRS += translation

for(subdir, SUBDIRS) {
    $${subdir}.depends += opennn
}

win32 {
    DEFINES += _HAS_STD_BYTE=0
    DEFINES += WIN32_LEAN_AND_MEAN
}

CUDA_PATH = $$(CUDA_PATH)
isEmpty(CUDA_PATH): CUDA_PATH = $$(CUDA_HOME)
win32: isEmpty(CUDA_PATH) {
    CUDA_BASE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    CUDA_VERSIONS_FOUND = $$files($$CUDA_BASE_DIR/v*, true)
    !isEmpty(CUDA_VERSIONS_FOUND): CUDA_PATH = $$last(CUDA_VERSIONS_FOUND)
}

if(!isEmpty(CUDA_PATH)) {
    CUDA_PATH = $$clean_path($$CUDA_PATH)
    CUDA_INCLUDE_PATH = $$CUDA_PATH/include

    for(subdir, SUBDIRS) {
        $${subdir}.includePath += $$CUDA_INCLUDE_PATH
    }
}

# OpenMP library
include(../opennmp.pri)
