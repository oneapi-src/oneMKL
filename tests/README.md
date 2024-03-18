# oneMKL Interfaces Testing

## Overview
Inside the `unit_tests` directory, there are domain-level directories which contain domain-specific tests, usually per function or per configuration.

## Steps
Functional testing is enabled by default, so all relevant functional tests will run automatically after the project is built successfully.

*Note: A set of `build options` define a `build configuration`. `CMake` builds and runs different set of tests depending on your `build configuration`. This is because `CMake` generates an export header file (config.hpp) for the selected build configuration. Check `<path to onemkl>/src/config.hpp.in` and `<path to onemkl>/src/CMakeLists.txt` for details. For details on how `CMake` performs export header generation, refer to [CMake documentation](https://cmake.org/cmake/help/v3.13/module/GenerateExportHeader.html).*

You can re-run tests without re-building the entire project.

#### The `CMake` Approach Works for any Generator
```bash
cmake --build . --target test
```

#### To use Generator-specific Commands:

```bash
# For ninja
ninja test
```

```bash
# For GNU Makefiles
ctest
# Test filter use case - runs only Gpu specific tests
ctest -R Gpu
# Exclude filtering use case - excludes Cpu tests
ctest -E Cpu
```

For more `ctest` options, refer to [ctest manual page](https://cmake.org/cmake/help/v3.13/manual/ctest.1.html).

## BLAS

The tests in the level\<x> directories are for the corresponding level\<x> BLAS routines. [GoogleTest](https://github.com/google/googletest) is used as the unit-testing framework.


*Refer to `<path to onemkl>/deps/googletest/LICENSE` for GoogleTest license.*
