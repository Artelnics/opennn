###################################################################################################
#                                                                                                 #
#   OpenNN: Open Neural Networks Library                                                          #
#   www.opennn.net                                                                                #
#                                                                                                 #
#   O P E N N N   P R O J E C T                                                                   #
#                                                                                                 #
#   Artificial Intelligence Techniques SL (Artelnics)                                             #
#   artelnics@artelnics.com                                                                       #
#                                                                                                 #
###################################################################################################

# CONFIGURATION

TEMPLATE = subdirs

CONFIG += ordered

CONFIG(release, debug|release) {
DEFINES += NDEBUG
}

SUBDIRS += opennn
SUBDIRS += examples
SUBDIRS += tests

