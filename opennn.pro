#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

# CONFIGURATION

TEMPLATE = subdirs

CONFIG(release, debug|release) {
DEFINES += NDEBUG
}

SUBDIRS += opennn
SUBDIRS += examples
SUBDIRS += blank
#SUBDIRS += tests

CONFIG += ordered

CUDA_PATH = $$(CUDA_PATH)
isEmpty(CUDA_PATH): CUDA_PATH = $$(CUDA_HOME)
win32: isEmpty(CUDA_PATH) {
    CUDA_BASE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
    CUDA_VERSIONS_FOUND = $$files($$CUDA_BASE_DIR/v*, true)
    !isEmpty(CUDA_VERSIONS_FOUND): CUDA_PATH = $$last(CUDA_VERSIONS_FOUND)
}

if(!isEmpty(CUDA_PATH)) {
    SUBDIRS += blank_cuda
}

message("[ROOT] Condfig finished. Projects: $$SUBDIRS")
