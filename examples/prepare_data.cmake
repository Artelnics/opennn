# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

# Stage only the selected example's data beside its executable. ZIP members keep
# the same relative names as the original loose assets; no Python is required.
if(NOT DEFINED OPENNN_DATA_SOURCE OR NOT DEFINED OPENNN_DATA_DESTINATION)
    message(FATAL_ERROR "Example data source and destination are required")
endif()

file(MAKE_DIRECTORY "${OPENNN_DATA_DESTINATION}")
file(GLOB entries "${OPENNN_DATA_SOURCE}/*")
foreach(entry IN LISTS entries)
    if(entry MATCHES "\\.zip$")
        # ImageDataset includes image timestamps in its generated cache signature.
        file(ARCHIVE_EXTRACT INPUT "${entry}" DESTINATION "${OPENNN_DATA_DESTINATION}" TOUCH)
    else()
        file(COPY "${entry}" DESTINATION "${OPENNN_DATA_DESTINATION}")
    endif()
endforeach()
