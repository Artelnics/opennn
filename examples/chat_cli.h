//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E X A M P L E   C H A T   C L I

#pragma once

#include <iostream>
#include <string>

#include "opennn/chat.h"

namespace opennn::examples
{

inline void run_chat_repl(ChatSession& session, const ChatOptions& options = {})
{
    std::cout << "Enter prompts. Empty line, 'exit' or 'quit' finishes.\n";

    std::string prompt;
    while (true)
    {
        std::cout << "\n> " << std::flush;
        if (!std::getline(std::cin, prompt)
            || prompt.empty()
            || prompt == "exit"
            || prompt == "quit")
            break;

        bool reasoning_started = false;
        bool content_started = false;
        const ChatResponse response = session.send(
            prompt, options,
            [&](const ChatDelta& delta)
            {
                if (delta.channel == GenerationChannel::Reasoning)
                {
                    if (!reasoning_started)
                    {
                        std::cout << "Thinking: ";
                        reasoning_started = true;
                    }
                }
                else if (!content_started)
                {
                    if (reasoning_started) std::cout << "\n";
                    std::cout << "Response: ";
                    content_started = true;
                }
                std::cout << delta.text << std::flush;
            });

        if (!content_started)
        {
            if (reasoning_started) std::cout << "\n";
            std::cout << "Response: " << response.content;
        }
        std::cout << "\n";
    }
    std::cout << "Bye!\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
