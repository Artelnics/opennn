//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B E R T   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tabular_dataset.h"

namespace opennn
{

class BertDataset : public TabularDataset
{
public:

    BertDataset(const filesystem::path& text_file,
                const filesystem::path& vocabulary_file,
                Index sequence_length);

    Index get_sequence_length() const { return sequence_length; }

private:

    Index sequence_length = 0;

    void write_tokens_csv(const filesystem::path& text_file,
                          const filesystem::path& vocabulary_file,
                          const filesystem::path& csv_path) const;

    void configure_bert_roles();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
