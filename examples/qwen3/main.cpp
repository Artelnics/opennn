//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q W E N 3   C H A T   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

//   Downloads a pretrained Qwen3-4B model and starts an interactive chat.

#include <filesystem>
#include <iostream>
#include <memory>

#include "opennn/neural_network/chat.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. Qwen3 chat." << endl;

        const filesystem::path data_directory = argc > 1 ? argv[1] : "../data";

        auto model = Qwen3::from_pretrained(
            Qwen3::Variant::B4, data_directory);
        Qwen3Tokenizer tokenizer(data_directory);

        ChatSession session(
            *model, tokenizer, make_unique<Qwen3ChatTemplate>());

        session.chat();
        return 0;
    }
    catch (const exception& e)
    {
        cout << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
