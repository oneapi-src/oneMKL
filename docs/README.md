### oneMKL documentation

This folder contains oneMKL documentation in reStructuredText (rST) format.

The documentation build step is skipped by default.
To enable building documentation from the main build:
- Set `-o build_doc=True` when building with Conan. For more information see [Building with Conan](../README.md#building-with-conan)
- Set `-DBUILD_DOC=ON` when building with CMake. For more information see [Building with CMake](../README.md#building-with-cmake)

To build documentation only use following cmake command from the current folder:
```bash
# Inside <path to onemkl>/docs
mkdir build && cd build
cmake ..
cmake --build .
```
Generated documentation can be found in `<path to onemkl>/docs/build/Documentation`
