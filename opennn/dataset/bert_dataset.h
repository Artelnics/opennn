// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/dataset/tabular_dataset.h"

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

    bool load_cache(const filesystem::path&);
    void build(const filesystem::path&, const filesystem::path&);
    void save_cache(const filesystem::path&, const vector<string>&) const;
    void configure(const vector<string>&, Index);
};

}
