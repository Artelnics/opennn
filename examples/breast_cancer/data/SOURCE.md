# Breast Cancer Wisconsin (Original) data

Wolberg, W. (1990), *Breast Cancer Wisconsin (Original)*.
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original),
DOI [10.24432/C5HP4Z](https://doi.org/10.24432/C5HP4Z).

Data licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Preserve the attribution, source and licence link when redistributing.

OpenNN adaptation: remove the ID column, omit rows containing `?`, map class
labels 2/4 to 0/1, add headers and use semicolons. All 683 cleaned rows match
that transformation, verified 2026-09-08.

The missing-values variant replaces these cells with `NA` (one-based data row
and column, excluding the header): (10,5), (90,4), (98,4), (101,5), (102,7),
(104,9), (105,2), (137,9), (159,3), (202,2), (209,7). All other cells match
the cleaned CSV. These missing values were introduced in the OpenNN example.
See the repository's DATASETS.md for download checksums and review.
