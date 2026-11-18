// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once
#include <fstream>
#include <iostream>
#include <string>

// HOST_BASELINE_FIELD of tools/common.py: this process's anonymous resident
// memory in MiB, the reading run.py polls for a CPU launch and subtracts this
// baseline from. Every driver, in every engine, reports it at the same point
// -- the engine configured, no data loaded and no model built -- so the
// workload figure is what the data, model and run add. Nothing is printed
// where procfs lacks the reading: run.py then keeps the whole-process peak
// with its note rather than subtracting a made-up zero.
inline void report_anonymous_baseline()
{
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind("RssAnon:", 0) == 0)
        {
            std::cout << "baseline_anonymous_rss_mib=" << std::stod(line.substr(8)) / 1024.0 << "\n";
            return;
        }
#endif
}
