### oneMKL documentation

This folder contains oneMKL documentation in reStructuredText (rST) format.

The documentation build step is skipped by default.
To enable building documentation from the main build, set `-DBUILD_DOC=ON`.
For more information see [Building with CMake](../README.md#building-with-cmake).

To build documentation only, use the following commands from the current folder:
```bash
# Inside <path to onemkl>/docs
mkdir build && cd build
cmake ..
cmake --build .
```
Generated documentation can be found in `<path to onemkl>/docs/build/Documentation`
