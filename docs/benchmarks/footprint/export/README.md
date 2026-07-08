# Standalone Export Benchmark

Purpose: demonstrate that OpenNN can export a trained model as standalone source
code (C, Python, JavaScript, PHP) instead of a runtime-dependent model file.

Compile `opennn_export.cpp` against the OpenNN library and run it; it trains a
small model and writes it out as source in each target language. Archive the
generated source plus a compile/run check of it as the result.
