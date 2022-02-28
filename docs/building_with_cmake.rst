.. _building_with_cmake:

Building with CMake
===================


#. 
   Make sure you have completed `Build Setup <#build-setup>`_. 

#. 
   Build and install all required `dependencies <#software-requirements>`_. 

Building for oneMKL
^^^^^^^^^^^^^^^^^^^

* On Linux*

  .. code-block:: bash

     # Inside <path to onemkl>
     mkdir build && cd build
     cmake .. [-DCMAKE_CXX_COMPILER=<path_to_dpcpp_compiler>/bin/dpcpp]  # required only if dpcpp is not found in environment variable PATH
              [-DCMAKE_C_COMPILER=<path_to_icx_compiler>/bin/icx]        # required only if icx is not found in environment variable PATH
              [-DMKL_ROOT=<mkl_install_prefix>]                          # required only if environment variable MKLROOT is not set
              [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]          # required only for testing
              [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>]      # required only for testing
     cmake --build .
     ctest
     cmake --install . --prefix <path_to_install_dir>

* On Windows*

  .. code-block:: bash

     # Inside <path to onemkl>
     md build && cd build
     cmake .. -G Ninja [-DCMAKE_CXX_COMPILER=<path_to_dpcpp_compiler>\bin\dpcpp]  # required only if dpcpp is not found in environment variable PATH
                       [-DCMAKE_C_COMPILER=<path_to_icx_compiler>\bin\icx]        # required only if icx is not found in environment variable PATH
                       [-DMKL_ROOT=<mkl_install_prefix>]                          # required only if environment variable MKLROOT is not set
                       [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]          # required only for testing
                       [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>]      # required only for testing
     ninja 
     ctest
     cmake --install . --prefix <path_to_install_dir>

Building for CUDA
^^^^^^^^^^^^^^^^^

* On Linux*

With the cuBLAS backend:

.. code-block:: bash

   # Inside <path to onemkl>
   mkdir build && cd build
   cmake .. [-DCMAKE_CXX_COMPILER=<path_to_clang++_compiler>/bin/clang++]  # required only if clang++ is not found in environment variable PATH
            [-DCMAKE_C_COMPILER=<path_to_clang_compiler>/bin/clang]        # required only if clang is not found in environment variable PATH
            -DENABLE_CUBLAS_BACKEND=True  \
            -DENABLE_MKLCPU_BACKEND=False                                  # disable Intel MKL CPU backend
            -DENABLE_MKLGPU_BACKEND=False                                  # disable Intel MKL GPU backend
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]              # required only for testing
   cmake --build .
   ctest
   cmake --install . --prefix <path_to_install_dir>

To build with the cuRAND backend instead simply replace:

.. code-block:: bash

   -DENABLE_CUBLAS_BACKEND=True   \

With:

.. code-block:: bash

   -DENABLE_CURAND_BACKEND=True   \

Building for ROCm (with hipSYCL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the AMD rocBLAS backend:


* On Linux*

.. code-block:: bash

   # Inside <path to onemkl>
   mkdir build && cd build
   cmake .. -DENABLE_CUBLAS_BACKEND=False                     \
            -DENABLE_MKLCPU_BACKEND=False/True                  # hipSYCL supports MKLCPU backend     
            -DENABLE_NETLIB_BACKEND=False/True                  # hipSYCL supports NETLIB backend
            -DENABLE_MKLGPU_BACKEND=False                       # disable Intel MKL GPU backend
            -DENABLE_ROCBLAS_BACKEND=True                     \
            -DTARGET_DOMAINS=blas                               # hipSYCL only supports the BLAS domain
            -DHIPSYCL_TARGETS=omp\;hip:gfx906                   # Specify the targetted device architectures 
            -DONEMKL_SYCL_IMPLEMENTATION=hipSYCL                # Use the hipSYCL cmake integration
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]   # required only for testing
   cmake --build .
   ctest
   cmake --install . --prefix <path_to_install_dir>

**AMD GPU device architectures**  

The device architecture can be retrieved via the ``rocminfo`` tool. The architecture will be displayed in the ``Name:`` row.

A few often-used architectures are listed below:

.. list-table::
   :header-rows: 1

   * - Architecture
     - AMD GPU name
   * - gfx906
     - | AMD Radeon Instinct(TM) MI50/60 Accelerator
       | AMD Radeon(TM) (Pro) VII Graphics Card
   * - gfx908
     - AMD Instinct(TM) MI 100 Accelerator
   * - gfx900
     - | Radeon Instinct(TM) MI 25 Accelerator
       | Radeon(TM) RX Vega 64/56 Graphics

Build Options
^^^^^^^^^^^^^

When building oneMKL the SYCL implementation can be determined, by setting the
``ONEMKL_SYCL_IMPLEMENTATION`` option. Possible values are:

* ``dpc++`` (default) for the
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_
  and for the ``clang++`` from
  `Intel project for LLVM* technology <https://github.com/intel/llvm/releases>`_ compilers.
* ``hipsycl`` for the `hipSYCL <https://github.com/illuhad/hipSYCL>`_ SYCL implementation.

All options specified in the Conan section are available to CMake. You can
specify these options using ``-D<cmake_option>=<value>``.

The following table provides a detailed mapping of options between Conan and
CMake.

.. list-table::
   :header-rows: 1

   * - Conan Option
     - CMake Option
     - Supported Values
     - Default Value 
   * - build_shared_libs
     - BUILD_SHARED_LIBS
     - True, False
     - True      
   * - enable_mklcpu_backend
     - ENABLE_MKLCPU_BACKEND
     - True, False
     - True      
   * - enable_mklgpu_backend
     - ENABLE_MKLGPU_BACKEND
     - True, False
     - True      
   * - *Not Supported*
     - ENABLE_CUBLAS_BACKEND
     - True, False
     - False     
   * - *Not Supported*
     - ENABLE_CUSOLVER_BACKEND
     - True, False
     - False     
   * - *Not Supported*
     - ENABLE_CURAND_BACKEND
     - True, False
     - False     
   * - *Not Supported*
     - ENABLE_NETLIB_BACKEND
     - True, False
     - False     
   * - *Not Supported*
     - ENABLE_ROCBLAS_BACKEND
     - True, False
     - False     
   * - enable_mklcpu_thread_tbb
     - ENABLE_MKLCPU_THREAD_TBB
     - True, False
     - True      
   * - build_functional_tests
     - BUILD_FUNCTIONAL_TESTS
     - True, False
     - True      
   * - build_doc
     - BUILD_DOC
     - True, False
     - False     
   * - target_domains (list)
     - TARGET_DOMAINS (list)
     - blas, lapack, rng
     - All domains 

.. note::
  ``build_functional_tests`` and related CMake options affect all domains at a
  global scope.

 
.. note::
  When building with hipSYCL, you must additionally provide
  ``-DHIPSYCL_TARGETS`` according to the targeted hardware. For the options,
  see the tables in the hipSYCL-specific sections.
