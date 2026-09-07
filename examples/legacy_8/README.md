# Preserved 8.x examples

These files were recovered from `master` at `efd566b38` during the 9.0
reconciliation. They are historical reference material, **not supported 9.x
CMake targets**. The source still uses the flat 8.x headers and XML-era APIs.
Use that historical commit in a separate checkout to inspect its original
layout. Do not point these project files at the 9.x library.

- `n2o_forecast`: action-conditioned wastewater forecasting; bundled CSV
  retained pending verification of its transformations and attribution.
- `wwt_optimization`: requires `WWTP_PO4_NH4_removal.csv`, which was not present
  in the merged master tree. The results file is not a replacement dataset.
- `forecasting`: Madrid NO2 binary artifacts retained for provenance review;
  their original producer and binary layout are not established here.

For maintained code, see `../forecasting_tinyml`, the time-series dataset
tests, and the response-optimization integration scenarios. A port of these
historical applications must explicitly translate window indices and verify
predictions before transferring their optimization settings. See
`../../MIGRATION.md` and `../../DATASETS.md`.
