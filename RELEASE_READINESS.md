# OpenNN 9.0.0 release candidate

The CMake project, shared-library major version, package compatibility and
consumer smoke project now target **9.0.0**. The changelog labels it an
unreleased candidate. No final release or tag is implied.

## Reconciliation and migration

Master through `efd566b38` is reconciled with development `5840396e9`.
`MERGE_RECONCILIATION.md` records all 98 conflicted paths and behavioral
choices. `MIGRATION.md` documents header/API changes, JSON and parameter
migration limitations, time-series indexing and genetic initialization.
Master-only legacy applications are preserved and clearly marked as 8.x
reference material; they are not supported 9.x build targets.

Neural Designer validation is **deferred at the owner's request**, because
the current application is unavailable here. It is not a gate for completing
this repository reconciliation and is not claimed as verified compatibility.
No public API was removed based on absence of local callers.

## Publication gates

- Hosted GCC, Clang and Windows CPU tests, package-consumer checks, CUDA
  compilation and ASan/UBSan must pass on the candidate commit.
- Linux GPU verification must pass unit and response-optimization integration
  suites. Historical results are not substitutes for the candidate's results.
- `python tools/check_dataset_manifest.py` must pass. The stronger `--release`
  check must also pass before publishing an archive containing all data.
- `python tools/check_release_scope.py --package-kind binary` must pass for the
  CMake/CPack package. A full-source archive remains blocked until every tracked
  asset's redistribution record is cleared.
- Resolve the data/derivative permission records listed in `DATASETS.md`, or
  provide reviewed reproducible replacements. Existing data is retained.
- If a complete production 8.x model becomes available, convert and validate it
  using the migration procedure. None exists in the repository or elsewhere, so
  9.0 explicitly makes no historical model-compatibility claim; this limitation
  is enforced by `RELEASE_SCOPE.json` rather than represented as completed work.
- Confirm the changelog describes the final contents, then create an annotated
  `v9.0.0` tag on the reviewed master commit and publish matching artifacts.

The provenance inventory is implemented; dataset clearance remains incomplete.
Seven source-data reconstruction recipes and repeatable Iris/concrete reference
model recipes are available in `tools/REPRODUCTION.md`. Candidate installation
archives and their validation procedure are documented in `tools/PACKAGING.md`.
In particular, identified upstream licences do not establish undocumented
local image sources, model-generation records or derivative transformations.
GitHub automatic source archives contain the tracked datasets even though
CMake's installed library package excludes them.

See `RELEASE_VERIFICATION.md` for actual validation evidence. Engineering
verification and publication clearance are separate statuses.
