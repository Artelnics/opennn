#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

TEMPLATE = subdirs

CONFIG(release, debug|release) {
    DEFINES += NDEBUG
}

SUBDIRS += opennn
SUBDIRS += examples
SUBDIRS += blank
SUBDIRS += tests

blank.depends = opennn
examples.depends = opennn
tests.depends = opennn

CONFIG += ordered

include(cuda.pri)

if($$CUDA_ENABLED)
{
    SUBDIRS += blank_cuda
    blank_cuda.depends = opennn
}

message("[ROOT] Config finished. Projects: $$SUBDIRS")
