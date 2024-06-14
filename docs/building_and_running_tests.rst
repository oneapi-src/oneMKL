.. _building_and_running_tests:

Building and Running Tests
==========================

The functional are tests are enabled by default, and can be enabled/disabled
with the CMake build parameter ``-DBUILD_FUNCTIONAL_TESTS=True/False``. Only tests
relevant for the enabled backends and target domains are built.

Building tests for some domains may require additional libraries for reference.

* BLAS: Requires a reference BLAS library.
* LAPACK: Requires a reference LAPACK library.

For both BLAS and LAPACK, shared libraries supporting both 32 and 64 bit
indexing are required.

A reference LAPACK implementation (including BLAS) can be built as the
following:

.. code-block:: bash

  git clone https://github.com/Reference-LAPACK/lapack.git 
  cd lapack; mkdir -p build; cd build 
  cmake -DCMAKE_INSTALL_PREFIX=~/lapack -DCBLAS=True -DLAPACK=True -DLAPACKE=True -DBUILD_INDEX64=True -DBUILD_SHARED_LIBS=True .. 
  cmake --build . -j --target install 
  cmake -DCMAKE_INSTALL_PREFIX=~/lapack -DCBLAS=True -DLAPACK=True -DLAPACKE=True -DBUILD_INDEX64=False -DBUILD_SHARED_LIBS=True .. 
  cmake --build . -j --target install

and then used in oneMKL by setting ``-REF_BLAS_ROOT=/path/to/lapack/install``
and ``-DREF_LAPACK_ROOT=/path/to/lapack/install``.

To run the tests, either use the CMake test driver, by running ``ctest``, or run
individual test binaries individually.

When running tests you may encounter the issue ``BACKEND NOT FOUND EXCEPTION``,
you may need to add your ``<oneMKL build directory>/lib`` to your
``LD_LIBRARY_PATH`` on Linux.
