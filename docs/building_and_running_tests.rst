.. _building_and_running_tests:

Building and Running Tests
==========================

The functional are tests are enabled by default, and can be enabled/disabled with
the CMake build parameter ``-DBUILD_FUNCTIONAL_TESTS=ON/OFF``. 
Only tests relevant for the enabled backends and target domains are built.

Building tests for some domains may require additional libraries for reference.

* BLAS: Requires a reference BLAS library.
* LAPACK: Requires a reference LAPACK library.

For both BLAS and LAPACK, shared libraries supporting both 32 and 64 bit indexing are required.

A reference LAPACK implementation (including BLAS) can be built as the following:

.. code-block:: bash

    git clone https://github.com/Reference-LAPACK/lapack.git 
    cd lapack; mkdir -p build; cd build 
    cmake -DCMAKE_INSTALL_PREFIX=~/lapack -DCBLAS=ON -DLAPACK=ON -DLAPACKE=ON -DBUILD_INDEX64=ON -DBUILD_SHARED_LIBS=ON .. 
    cmake --build . -j --target install 
    cmake -DCMAKE_INSTALL_PREFIX=~/lapack -DCBLAS=ON -DLAPACK=ON -DLAPACKE=ON -DBUILD_INDEX64=OFF -DBUILD_SHARED_LIBS=ON .. 
    cmake --build . -j --target install

and then used in oneMKL by setting ``-REF_BLAS_ROOT=/path/to/lapack/install`` and ``-DREF_LAPACK_ROOT=/path/to/lapack/install``.

To run the tests, either use the CMake test driver, by running ``ctest``, or run individual test binaries individually.

When running tests you may encounter the issue ``BACKEND NOT FOUND EXECEPTION``, you may need to add your ``<oneMKL build directory>/lib`` to your ``LD_LIBRARY_PATH`` on Linux.
