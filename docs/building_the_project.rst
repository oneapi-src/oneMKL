.. _building_the_project:

Building the Project
====================

.. _build_setup:

Build Setup
###########

#. 
   Install Intel(R) oneAPI DPC++ Compiler (select the variant as described in
   :ref:`Selecting a Compiler<selecting_a_compiler>`).

#. 
   Clone this project to ``<path to onemkl>``\ , where ``<path to onemkl>``
   is the root directory of this repository.

#. 
   Download and install the required dependencies manually and :ref:`Build with CMake <building_with_cmake>`.


.. _build_setup_with_hipsycl:

Build Setup with hipSYCL
########################

#. 
   Make sure that the dependencies of hipSYCL are fulfilled. For a detailed
   description, see the
   `hipSYCL installation readme <https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md#software-dependencies>`_.

#. 
   Install hipSYCL with the prefered backends enabled. hipSYCL supports
   various backends. You can customize support for the target system at
   compile time by setting the appropriate configuration flags; see the
   `hipSYCL documentation <https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md>`_
   for instructions.

#. 
   Install `AMD rocBLAS <https://rocblas.readthedocs.io/en/master/install.html>`_.

#. 
   Clone this project to ``<path to onemkl>``, where ``<path to onemkl>`` is
   the root directory of this repository.

#. 
   Download and install the required dependencies manually and
   :ref:`Build with CMake <building_with_cmake>`.



.. _building_with_cmake:

Building with CMake
###################

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
     cmake .. [-DCMAKE_CXX_COMPILER=<path_to_icpx_compiler>/bin/icpx]    # required only if icpx is not found in environment variable PATH
              [-DCMAKE_C_COMPILER=<path_to_icx_compiler>/bin/icx]        # required only if icx is not found in environment variable PATH
              [-DMKL_ROOT=<mkl_install_prefix>]                          # required only if environment variable MKLROOT is not set
              [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]          # required only for testing
              [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>]      # required only for testing
     cmake --build .
     ctest
     cmake --install . --prefix <path_to_install_dir>                    # required to have full package structure

* On Windows*

  .. code-block:: bash

     # Inside <path to onemkl>
     md build && cd build
     cmake .. -G Ninja [-DCMAKE_CXX_COMPILER=<path_to_icx_compiler>\bin\icx]      # required only if icx is not found in environment variable PATH
                       [-DCMAKE_C_COMPILER=<path_to_icx_compiler>\bin\icx]        # required only if icx is not found in environment variable PATH
                       [-DMKL_ROOT=<mkl_install_prefix>]                          # required only if environment variable MKLROOT is not set
                       [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]          # required only for testing
                       [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>]      # required only for testing
     ninja 
     ctest
     cmake --install . --prefix <path_to_install_dir>                             # required to have full package structure

Building for CUDA (with hipSYCL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* On Linux*

With the cuBLAS backend:

.. code-block:: bash

   # Inside <path to onemkl>
   mkdir build && cd build
   cmake .. -DENABLE_CUBLAS_BACKEND=True \
            -DENABLE_MKLGPU_BACKEND=False                                # Disable all backends except for cuBLAS
            -DENABLE_MKLCPU_BACKEND=False \
            -DENABLE_NETLIB_BACKEND=False \
            -DENABLE_ROCBLAS_BACKEND=False \
            -DHIPSYCL_TARGETS=cuda:sm_75 \                               # Specify the targeted device architectures 
            -DONEMKL_SYCL_IMPLEMENTATION=hipSYCL \
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]            # required only for testing
   cmake --build .
   ctest
   cmake --install . --prefix <path_to_install_dir>                      # required to have full package structure

To build with the cuRAND backend instead simply replace:

.. code-block:: bash

   -DENABLE_CUBLAS_BACKEND=True   \

With:

.. code-block:: bash

   -DENABLE_CURAND_BACKEND=True   \


Building for CUDA (with clang++)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   cmake --install . --prefix <path_to_install_dir>                        # required to have full package structure


The CuFFT and CuRAND backends can be enabled in a similar way to the CuBLAS backend, by setting the corresponding CMake variable(s) to `True`:

.. code-block:: bash

   -DENABLE_CUFFT_BACKEND=True    \
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
            -DTARGET_DOMAINS=blas                               # hipSYCL supports BLAS and RNG domains
            -DHIPSYCL_TARGETS=omp\;hip:gfx906                   # Specify the targetted device architectures 
            -DONEMKL_SYCL_IMPLEMENTATION=hipSYCL                # Use the hipSYCL cmake integration
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]   # required only for testing
   cmake --build .
   ctest
   cmake --install . --prefix <path_to_install_dir>             # required to have full package structure

To build with the rocRAND backend instead simply replace:

.. code-block:: bash

   -DENABLE_ROCBLAS_BACKEND=True   \
   -DTARGET_DOMAINS=blas

With:

.. code-block:: bash

   -DENABLE_ROCRAND_BACKEND=True   \
   -DTARGET_DOMAINS=rng

Building for ROCm (with clang++)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the AMD rocBLAS backend:


* On Linux*

.. code-block:: bash

   # Inside <path to onemkl>
   mkdir build && cd build
   cmake .. [-DCMAKE_CXX_COMPILER=<path_to_clang++_compiler>/bin/clang++]  # required only if clang++ is not found in environment variable PATH
            [-DCMAKE_C_COMPILER=<path_to_clang_compiler>/bin/clang]        # required only if clang is not found in environment variable PATH
            -DENABLE_CUBLAS_BACKEND=False                                \
            -DENABLE_MKLCPU_BACKEND=False                                \ # disable Intel MKL CPU backend
            -DENABLE_MKLGPU_BACKEND=False                                \ # disable Intel MKL GPU backend
            -DENABLE_ROCBLAS_BACKEND=True                                \
            -DHIP_TARGETS=gfx90a                                         \ # Specify the targetted device architectures
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]              # required only for testing
   cmake --build .
   export ONEAPI_DEVICE_SELECTOR="hip:gpu"
   ctest
   cmake --install . --prefix <path_to_install_dir>                        # required to have full package structure

The rocRAND, rocFFT, and rocSOLVER backends can be enabled in a similar way to the rocBLAS backend, by setting the corresponding CMake variable(s) to `True`:

.. code-block:: bash

   -DENABLE_ROCRAND_BACKEND=True     \
   -DENABLE_ROCFFT_BACKEND=True      \
   -DENABLE_ROCSOLVER_BACKEND=True   \

**AMD GPU device architectures**  

The device architecture can be retrieved via the ``rocminfo`` tool. The architecture will be displayed in the ``Name:`` row.

A few often-used architectures are listed below:

.. list-table::
   :header-rows: 1

   * - Architecture
     - AMD GPU name
   * - gfx90a
     - AMD Instinct(TM) MI210/250/250X Accellerator
   * - gfx908
     - AMD Instinct(TM) MI 100 Accelerator
   * - gfx906
     - | AMD Radeon Instinct(TM) MI50/60 Accelerator
       | AMD Radeon(TM) (Pro) VII Graphics Card
   * - gfx900
     - | Radeon Instinct(TM) MI 25 Accelerator
       | Radeon(TM) RX Vega 64/56 Graphics

Building for portBLAS
^^^^^^^^^^^^^^^^^^^^^^

Note the portBLAS backend is experimental and currently only supports a
subset of the operations and features. The portBLAS backend cannot be enabled
with other backends and can only be used with the compile time dispatch.
The portBLAS backend uses the `portBLAS <https://github.com/codeplaysoftware/portBLAS>`_
project as a header-only library.

* On Linux*

.. code-block:: bash

   # Inside <path to onemkl>
   mkdir build && cd build
   cmake .. -DENABLE_PORTBLAS_BACKEND=ON \
            -DENABLE_MKLCPU_BACKEND=OFF  \
            -DENABLE_MKLGPU_BACKEND=OFF  \
            -DTARGET_DOMAINS=blas \
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>] \ # required only for testing
            [-DPORTBLAS_DIR=<path to portBLAS install directory>]
   cmake --build .
   ./bin/test_main_blas_ct
   cmake --install . --prefix <path_to_install_dir>


portBLAS will be downloaded automatically if not found.
By default, the portBLAS backend is not tuned for any specific device which
will impact performance.
portBLAS can be tuned for a specific hardware target by adding compiler
definitions in 2 ways:

#.
  Manually specify a tuning target with ``-DPORTBLAS_TUNING_TARGET=<target>``.
  The list of portBLAS targets can be found
  `here <https://github.com/codeplaysoftware/portBLAS#cmake-options>`_.
  This will automatically set ``-fsycl-targets`` if needed.
  In case of ``AMD_GPU`` target, it is mandatory to set one or more device
  architectures by means of ``HIP_TARGETS``, e.g., ``-DHIP_TARGETS=gfx90a``.
  In case of ``NVIDIA_GPU`` target, it is possible to select a specific device
  architecture by means of ``CUDA_TARGET``, e.g., ``-DCUDA_TARGET=sm_80``.
#.
  If one target is set via ``-fsycl-targets`` the configuration step will
  try to automatically detect the portBLAS tuning target. One can manually
  specify ``-fsycl-targets`` via ``CMAKE_CXX_FLAGS``. See
  `DPC++ User Manual <https://intel.github.io/llvm-docs/UsersManual.html>`_
  for more information on ``-fsycl-targets``.

Building for portFFT
^^^^^^^^^^^^^^^^^^^^^^

Note the portFFT backend is experimental and currently only supports a
subset of the operations and features.
The portFFT backend uses the `portFFT <https://github.com/codeplaysoftware/portFFT>`_
project as a header-only library.

* On Linux*

.. code-block:: bash

   # Inside <path to onemkl>
   mkdir build && cd build
   cmake .. -DENABLE_PORTFFT_BACKEND=ON \
            -DENABLE_MKLCPU_BACKEND=OFF  \
            -DENABLE_MKLGPU_BACKEND=OFF  \
            -DTARGET_DOMAINS=dft \
            [-DPORTFFT_REGISTERS_PER_WI=128] \ # Example portFFT tuning parameter
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>] \ # required only for testing
            [-DPORTFFT_DIR=<path to portFFT install directory>]
   cmake --build .
   ./bin/test_main_dft_ct
   cmake --install . --prefix <path_to_install_dir>


portFFT will be downloaded automatically if not found.

By default, the portFFT backend is not tuned for any specific device. The tuning flags are
detailed in the `portFFT <https://github.com/codeplaysoftware/portFFT>`_ repository.
The tuning parameters can be set at configuration time,
with the above example showing how to set the tuning parameter
``PORTFFT_REGISTERS_PER_WI``. Note that some tuning configurations may be incompatible
with some targets.

The portFFT library is compiled using the same ``-fsycl-targets`` as specified
by the ``CMAKE_CXX_FLAGS``. If none are found, it will compile for
``-fsycl-targets=nvptx64-nvidia-cuda,spir64``. To enable HIP targets,
``HIP_TARGETS`` must be specified. See
`DPC++ User Manual <https://intel.github.io/llvm-docs/UsersManual.html>`_
for more information on ``-fsycl-targets``.


Build Options
^^^^^^^^^^^^^

When building oneMKL the SYCL implementation can be specified by setting the
``ONEMKL_SYCL_IMPLEMENTATION`` option. Possible values are:

* ``dpc++`` (default) for the
  `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_
  and for the
  `oneAPI DPC++ Compiler <https://github.com/intel/llvm>`_ compilers.
* ``hipsycl`` for the `hipSYCL <https://github.com/illuhad/hipSYCL>`_ SYCL implementation.

The following table provides details of CMake options and their default values:

.. list-table::
   :header-rows: 1

   * - CMake Option
     - Supported Values
     - Default Value 
   * - BUILD_SHARED_LIBS
     - True, False
     - True      
   * - ENABLE_MKLCPU_BACKEND
     - True, False
     - True      
   * - ENABLE_MKLGPU_BACKEND
     - True, False
     - True      
   * - ENABLE_CUBLAS_BACKEND
     - True, False
     - False     
   * - ENABLE_CUSOLVER_BACKEND
     - True, False
     - False     
   * - ENABLE_CUFFT_BACKEND
     - True, False
     - False     
   * - ENABLE_CURAND_BACKEND
     - True, False
     - False     
   * - ENABLE_NETLIB_BACKEND
     - True, False
     - False     
   * - ENABLE_ROCBLAS_BACKEND
     - True, False
     - False     
   * - ENABLE_ROCFFT_BACKEND
     - True, False
     - False    
   * - ENABLE_MKLCPU_THREAD_TBB
     - True, False
     - True      
   * - ENABLE_PORTBLAS_BACKEND
     - True, False
     - False      
   * - ENABLE_PORTFFT_BACKEND
     - True, False
     - False      
   * - BUILD_FUNCTIONAL_TESTS
     - True, False
     - True      
   * - BUILD_EXAMPLES
     - True, False
     - True      
   * - BUILD_DOC
     - True, False
     - False     
   * - TARGET_DOMAINS (list)
     - blas, lapack, rng, dft
     - All domains 

.. note::
  ``build_functional_tests`` and related CMake options affect all domains at a
  global scope.

 
.. note::
  When building with hipSYCL, you must additionally provide
  ``-DHIPSYCL_TARGETS`` according to the targeted hardware. For the options,
  see the tables in the hipSYCL-specific sections.


.. note::
  When building with clang++ for AMD backends, you must additionally set
  ``ONEAPI_DEVICE_SELECTOR`` to ``hip:gpu`` and provide ``-DHIP_TARGETS`` according to
  the targeted hardware. This backend has only been tested for the ``gfx90a``
  architecture (MI210) at the time of writing.

.. note::
  When building with ``BUILD_FUNCTIONAL_TESTS=yes`` (default option) only single CUDA backend can be built
  (`#270 <https://github.com/oneapi-src/oneMKL/issues/270>`_).

.. _project_cleanup:

Project Cleanup
###############

Most use-cases involve building the project without the need to cleanup the
build directory. However, if you wish to cleanup the build directory, you can
delete the ``build`` folder and create a new one. If you wish to cleanup the
build files but retain the build configuration, following commands will help
you do so.

.. code-block:: sh

   # If you use "GNU/Unix Makefiles" for building,
   make clean

   # If you use "Ninja" for building
   ninja -t clean
