# Candidate installation archives

CPack produces installation packages from a completed OpenNN build. Configure
with `OpenNN_INSTALL=ON` (the default), build the library, and keep package
output outside the checkout:

```sh
cpack --config /absolute/path/to/build/CPackConfig.cmake -C Release -G ZIP -B ../candidate-packages
cpack --config /absolute/path/to/build/CPackConfig.cmake -C Release -G TGZ -B ../candidate-packages
```

The archive name contains `9.0.0-candidate`, the system, target processor and
CPU/CUDA backend. Each archive has a SHA-256 companion file. Its installed
`share/doc/OpenNN/build-info.json` records the library version, compiler/version,
configuration, CUDA, shared-library and LTO settings. The installation includes
OpenNN's LGPL/GPL licence texts, release/migration documentation, libjpeg-turbo
notices, zlib's licence, and fetched Eigen/cuDNN frontend notices when applicable.
Optional FlashAttention installations include its and its CUTLASS dependency's
licence notices as well.

Extract into a new directory, check notices, and configure a separate consumer:

```sh
python tools/check_installed_package.py /absolute/path/to/extracted-prefix
cmake -S tools/package_smoke -B ../archive-consumer -DCMAKE_PREFIX_PATH=/absolute/path/to/extracted-prefix -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
cmake --build ../archive-consumer --config Release
```

Run `../archive-consumer/opennn_package_smoke` (or the executable under
`Release/` on Visual Studio). Use a compatible compiler/toolchain for the binary
package. Fetched Eigen and zlib are installed with the library; an Eigen package
found externally during configuration remains an external consumer dependency.
OpenMP, oneTBB, oneDNN, MKL and NVIDIA runtimes are not automatically copied into
the archive. Supply any dependencies selected by that build. A package built
with LTO also requires the matching compiler/linker support.

These are binary installation archives. They exclude example datasets and model
assets. They do not certify the separate GitHub source archive's dataset rights,
historical model compatibility, or publication approval. Record the source commit,
archive checksums, toolchain and actual consumer results in RELEASE_VERIFICATION.md
before promotion. CPack does not create a Git tag or GitHub release.
