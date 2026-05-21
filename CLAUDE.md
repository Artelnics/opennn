# Code rules for opennn (C++)

1. Don't add comments.
2. Use explicit names for variables, max 3 words concatenated (e.g. `objectives_number`).
3. Use `i`, `j` for simple `for` counters.
4. Vectorize as much as possible using Eigen and `std` functions.
5. Simplify code — avoid deeply nested loops and conditions.
6. Keep formatting coherent with the surrounding code.
7. Don't create structs or classes unless explicitly asked.
8. Don't create new files unless explicitly asked.
9. Before adding a helper, search the rest of the library for an existing utility; if the helper fits another file's concept better, put it there.
10. Avoid adding dependencies, especially ones that mix procedural and OO concepts.
11. Instead of adding private members, prefer `get_*` accessors for computable objects and variables.
