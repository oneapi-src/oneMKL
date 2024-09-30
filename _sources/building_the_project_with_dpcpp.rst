.. _building_the_project_with_dpcpp:

Building the Project with DPC++
===============================

This page describes building the oneMKL Interfaces with either the Intel(R)
oneAPI DPC++ Compiler or open-source oneAPI DPC++ Compiler. For guidance on
building the project with AdaptiveCpp, see
:ref:`building_the_project_with_adaptivecpp`.

.. _build_setup_with_dpcpp:

Environment Setup
##################

#. 
   Install the required DPC++ compiler (Intel(R) DPC++ or Open DPC++ - see
   :ref:`Selecting a Compiler<selecting_a_compiler>`).

#. 
   Clone this project. The root directory of the cloned repository will be
   referred to as ``<path to onemkl>``.

#. 
   Build and install all `required dependencies
   <https://github.com/oneapi-src/oneMKL?tab=readme-ov-file#software-requirements>`_. 

.. _build_introduction_with_dpcpp:

Build Commands
###############

The build commands for various compilers and backends differ mostly in setting
the values of CMake options for compiler and backend. In this section, we
describe the common build commands. We will discuss backend-specific details in
the `Backends`_ section and provide examples in `CMake invocation examples`_.

On Linux, the common form of the build command looks as follows (see `Building
for Windows`_ for building on Windows):

.. code-block:: bash

  # Inside <path to onemkl>
  mkdir build && cd build
  cmake .. -DCMAKE_CXX_COMPILER=$CXX_COMPILER    \ # Should be icpx or clang++
          -DCMAKE_C_COMPILER=$C_COMPILER         \ # Should be icx or clang
          -DENABLE_MKLCPU_BACKEND=False          \ # Optional: The MKLCPU backend is True by default.
          -DENABLE_MKLGPU_BACKEND=False          \ # Optional: The MKLGPU backend is True by default.
          -DENABLE_<BACKEND_NAME>_BACKEND=True   \ # Enable any other backend(s) (optional)
          -DENABLE_<BACKEND_NAME_2>_BACKEND=True \ # Multiple backends can be enabled at once.
          -DBUILD_FUNCTIONAL_TESTS=False         \ # See page *Building and Running Tests* for more on building tests. True by default.
          -DBUILD_EXAMPLES=False                   # Optional: True by default.
  cmake --build .
  cmake --install . --prefix <path_to_install_dir>  # required to have full package structure

In the above, the ``$CXX_COMPILER`` and ``$C_COMPILER`` should be set to
``icpx`` and ``icx`` respectively when using the Intel(R) oneAPI DPC++ Compiler,
or ``clang++`` and ``clang`` respectively when using the Open DPC++ Compiler. 

Backends should be enabled by setting ``-DENABLE_<BACKEND_NAME>_BACKEND=True`` for
each desired backend. By default, only the ``MKLGPU`` and ``MKLCPU`` backends
are enabled. Multiple backends for multiple device vendors can be enabled at
once (albeit with limitations when using portBLAS and portFFT). The supported
backends for the compilers are given in the table at `oneMKL supported
configurations table
<https://github.com/oneapi-src/oneMKL?tab=readme-ov-file#supported-configurations>`_,
and the CMake option names are given in the table below. Some backends may
require additional parameters to be set. See the relevant section below for
additional guidance.

If a backend library supports multiple domains (i.e., BLAS, LAPACK, DFT, RNG,
sparse BLAS), it may be desirable to only enable selected domains. For this, the
``TARGET_DOMAINS`` variable should be set. See the section `TARGET_DOMAINS`_.

By default, the library also additionally builds examples and tests. These can
be disabled by setting the parameters ``BUILD_FUNCTIONAL_TESTS`` and
``BUILD_EXAMPLES`` to ``False``. Building the functional tests requires
additional external libraries for the BLAS and LAPACK domains. See the section
:ref:`building_and_running_tests` for more information.

The most important supported build options are:

.. list-table::
   :header-rows: 1

   * - CMake Option
     - Supported Values
     - Default Value 
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
   * - ENABLE_ROCSOLVER_BACKEND
     - True, False
     - False     
   * - ENABLE_ROCRAND_BACKEND
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
   * - TARGET_DOMAINS (list)
     - blas, lapack, rng, dft, sparse_blas
     - All domains 

Some additional build options are given in the section `Additional build options`_.

.. _build_target_domains:

TARGET_DOMAINS
^^^^^^^^^^^^^^

oneMKL supports multiple domains: BLAS, DFT, LAPACK, RNG and sparse BLAS. The
domains built by oneMKL can be selected using the ``TARGET_DOMAINS`` parameter.
In most cases, ``TARGET_DOMAINS`` is set automatically according to the domains
supported by the backend libraries enabled. However, while most backend
libraries support only one of these domains, but some may support multiple. For
example, the ``MKLCPU`` backend supports every domain. To enable support for
only the BLAS domain in the oneMKL Interfaces whilst compiling with ``MKLCPU``,
``TARGET_DOMAINS`` could be set to ``blas``. To enable BLAS and DFT,
``-DTARGET_DOMAINS="blas dft"`` would be used.


Backends
#########

.. _build_for_intel_onemkl_dpcpp:

Building for Intel(R) oneMKL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Intel(R) oneMKL backend supports multiple domains on both x86 CPUs and Intel
GPUs. The MKLCPU backend using Intel(R) oneMKL for x86 CPU is enabled by
default, and controlled with the parameter ``ENABLE_MKLCPU_BACKEND``. The MKLGPU
backend using Intel(R) oneMKL for Intel GPU is enabled by default, and
controlled with the parameter ``ENABLE_MKLGPU_BACKEND``.

When using the Intel(R) oneAPI DPC++ Compiler, it is likely that Intel(R) oneMKL
will be found automatically. If it is not, the parameter ``MKL_ROOT`` can be set
to point to the installation prefix of Intel(R) oneMKL. Alternatively, the
``MKLROOT`` environment variable can be set, either manually or by using an
environment script provided by the package.


.. _build_for_CUDA_dpcpp:

Building for CUDA
^^^^^^^^^^^^^^^^^

The CUDA backends can be enabled with ``ENABLE_CUBLAS_BACKEND``,
``ENABLE_CUFFT_BACKEND``, ``ENABLE_CURAND_BACKEND``, and
``ENABLE_CUSOLVER_BACKEND``.

No additional parameters are required for using CUDA libraries. In most cases,
the CUDA libraries should be found automatically by CMake.

.. _build_for_ROCM_dpcpp:

Building for ROCm
^^^^^^^^^^^^^^^^^

The ROCm backends can be enabled with ``ENABLE_ROCBLAS_BACKEND``,
``ENABLE_ROCFFT_BACKEND``, ``ENABLE_ROCSOLVER_BACKEND`` and
``ENABLE_ROCRAND_BACKEND``.

For *RocBLAS*, *RocSOLVER* and *RocRAND*, the target device architecture must be
set. This can be set with using the ``HIP_TARGETS`` parameter. For example, to
enable a build for MI200 series GPUs, ``-DHIP_TARGETS=gfx90a`` should be set.
Currently, DPC++ can only build for a single HIP target at a time. This may
change in future versions.

A few often-used architectures are listed below:

.. list-table::
   :header-rows: 1

   * - Architecture
     - AMD GPU name
   * - gfx90a
     - AMD Instinct(TM) MI210/250/250X Accelerator
   * - gfx908
     - AMD Instinct(TM) MI 100 Accelerator
   * - gfx906
     - | AMD Radeon Instinct(TM) MI50/60 Accelerator
       | AMD Radeon(TM) (Pro) VII Graphics Card
   * - gfx900
     - | Radeon Instinct(TM) MI 25 Accelerator
       | Radeon(TM) RX Vega 64/56 Graphics

For a host with ROCm installed, the device architecture can be retrieved via the
``rocminfo`` tool. The architecture will be displayed in the ``Name:`` row.

.. _build_for_other_SYCL_devices:

Building for other SYCL devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SYCL enables portable heterogeneous computing on a wide range of accelerators.
Consequently, it is possible to use oneMKL Interfaces with accelerators not
anticipated by the oneMKL Interfaces team.

For generic SYCL devices, only the portBLAS backend is enabled. The user must
set the appropriate ``-fsycl-targets`` for their device, and also any
``PORTBLAS_TUNING_TARGET`` required for performance. See
`Building for portBLAS`_. Extensive testing is strongly advised for these
unsupported configurations.

.. _build_for_portlibs_dpcpp:

Pure SYCL backends: portBLAS and portFFT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`portBLAS <https://github.com/codeplaysoftware/portBLAS>`_ and `portFFT
<https://github.com/codeplaysoftware/portFFT>`_ are experimental pure-SYCL
backends that work on all SYCL targets supported by the DPC++ compiler. Since
they support multiple targets, they cannot be enabled with other backends in the
same domain, or the MKLCPU or MKLGPU backends. Both libraries are experimental
and currently only support a subset of operations and features.

For best performance, both libraries must be tuned. See the individual sections
for more details.

Both portBLAS and portFFT are used as header-only libraries, and will be
downloaded automatically if not found.

.. _build_for_portblas_dpcpp:

Building for portBLAS
---------------------

`portBLAS <https://github.com/codeplaysoftware/portBLAS>`_ is
enabled by setting ``-DENABLE_PORTBLAS_BACKEND=True``.

By default, the portBLAS backend is not tuned for any specific device.
This tuning is required to achieve best performance.
portBLAS can be tuned for a specific hardware target by adding compiler
definitions in 2 ways:

#.
  Manually specify a tuning target with ``-DPORTBLAS_TUNING_TARGET=<target>``.
  The list of portBLAS targets can be found
  `here <https://github.com/codeplaysoftware/portBLAS#cmake-options>`_.
  This will automatically set ``-fsycl-targets`` if needed.
#.
  If one target is set via ``-fsycl-targets`` the configuration step will
  try to automatically detect the portBLAS tuning target. One can manually
  specify ``-fsycl-targets`` via ``CMAKE_CXX_FLAGS``. See
  `DPC++ User Manual <https://intel.github.io/llvm-docs/UsersManual.html>`_
  for more information on ``-fsycl-targets``.

portBLAS relies heavily on JIT compilation. This may cause time-outs on some
systems. To avoid this issue, use ahead-of-time compilation through tuning
targets or ``sycl-targets``.

.. _build_for_portfft_dpcpp:

Building for portFFT
---------------------

`portFFT <https://github.com/codeplaysoftware/portFFT>`_ is enabled by setting
``-DENABLE_PORTFFT_BACKEND=True``.

By default, the portFFT backend is not tuned for any specific device. The tuning
flags are detailed in the `portFFT
<https://github.com/codeplaysoftware/portFFT>`_ repository, and can set at
configuration time. Note that some tuning configurations may be incompatible
with some targets.

The portFFT library is compiled using the same ``-fsycl-targets`` as specified
by the ``CMAKE_CXX_FLAGS``. If none are found, it will compile for
``-fsycl-targets=spir64``, and -if the compiler supports it-
``nvptx64-nvidia-cuda``. To enable HIP targets, ``HIP_TARGETS`` must be
specified. See `DPC++ User Manual
<https://intel.github.io/llvm-docs/UsersManual.html>`_ for more information on
``-fsycl-targets``.

.. _build_additional_options_dpcpp:

Additional Build Options
##########################

When building oneMKL the SYCL implementation can be specified by setting the
``ONEMKL_SYCL_IMPLEMENTATION`` option. Possible values are:

* ``dpc++`` (default) for the `Intel(R) oneAPI DPC++ Compiler
  <https://software.intel.com/en-us/oneapi/dpc-compiler>`_ and for the `oneAPI
  DPC++ Compiler <https://github.com/intel/llvm>`_ compilers.
* ``hipsycl`` for the `AdaptiveCpp <https://github.com/illuhad/AdaptiveCpp>`_
  SYCL implementation.
Please see :ref:`building_the_project_with_adaptivecpp` if using this option.

The following table provides details of CMake options and their default values:

.. list-table::
   :header-rows: 1

   * - CMake Option
     - Supported Values
     - Default Value 
   * - BUILD_SHARED_LIBS
     - True, False
     - True      
   * - BUILD_DOC
     - True, False
     - False     


.. note::
  When building with ``clang++`` for AMD backends, you must additionally set
  ``ONEAPI_DEVICE_SELECTOR`` to ``hip:gpu`` and provide ``-DHIP_TARGETS`` 
  according to the targeted hardware. This backend has only been tested for the 
  ``gfx90a`` architecture (MI210) at the time of writing. 

.. note::
  When building with ``BUILD_FUNCTIONAL_TESTS=True`` (default option) only single CUDA backend can be built
  (`#270 <https://github.com/oneapi-src/oneMKL/issues/270>`_).


.. _build_invocation_examples_dpcpp:

CMake invocation examples
##########################

Build oneMKL with support for Nvidia GPUs with tests
disabled using the Ninja build system:

.. code-block:: bash

  cmake $ONEMKL_DIR \
      -GNinja \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      -DENABLE_MKLGPU_BACKEND=False \
      -DENABLE_MKLCPU_BACKEND=False \
      -DENABLE_CUFFT_BACKEND=True \
      -DENABLE_CUBLAS_BACKEND=True \
      -DENABLE_CUSOLVER_BACKEND=True \
      -DENABLE_CURAND_BACKEND=True \
      -DBUILD_FUNCTIONAL_TESTS=False

``$ONEMKL_DIR`` points at the oneMKL source directly. The x86 CPU (``MKLCPU``)
and Intel GPU (``MKLGPU``) backends are enabled by default, but are disabled
here. The backends for Nvidia GPUs must all be explicilty enabled. The tests are
disabled, but the examples will still be built.

Building oneMKL with support for AMD GPUs with tests
disabled:

.. code-block:: bash

  cmake $ONEMKL_DIR \
      -DCMAKE_CXX_COMPILER=clang++ \ 
      -DCMAKE_C_COMPILER=clang \
      -DENABLE_MKLCPU_BACKEND=False \ 
      -DENABLE_MKLGPU_BACKEND=False \
      -DENABLE_ROCFFT_BACKEND=True  \ 
      -DENABLE_ROCBLAS_BACKEND=True \
      -DENABLE_ROCSOLVER_BACKEND=True \ 
      -DHIP_TARGETS=gfx90a \
      -DBUILD_FUNCTIONAL_TESTS=False

``$ONEMKL_DIR`` points at the oneMKL source directly. The x86 CPU (``MKLCPU``)
and Intel GPU (``MKLGPU``) backends are enabled by default, but are disabled
here. The backends for AMD GPUs must all be explicilty enabled. The tests are
disabled, but the examples will still be built.


Build oneMKL for the DFT domain only with support for x86 CPU, Intel GPU, AMD
GPU and Nvidia GPU with testing enabled:

.. code-block:: bash

  cmake $ONEMKL_DIR \ 
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_C_COMPILER=icx \ 
      -DENABLE_ROCFFT_BACKEND=True \
      -DENABLE_CUFFT_BACKEND=True \
      -DTARGET_DOMAINS=dft \
      -DBUILD_EXAMPLES=False

Note that this is not a supported configuration, and requires Codeplay's oneAPI
for `AMD <https://developer.codeplay.com/products/oneapi/amd/home/>`_ and
`Nvidia <https://developer.codeplay.com/products/oneapi/nvidia/home/>`_ GPU
plugins. The MKLCPU and MKLGPU backends are enabled by
default, with backends for Nvidia GPU and AMD GPU explicitly enabled.
``-DTARGET_DOMAINS=dft`` causes only DFT backends to be built. If this was not
set, the backend libraries to enable the use of BLAS, LAPACK and RNG with MKLGPU
and MKLCPU would also be enabled. The build of examples is disabled. Since
functional testing was not disabled, tests would be built.

Build oneMKL for the BLAS domain on a generic SYCL device:

.. code-block:: bash

  cmake $ONEMKL_DIR \ 
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \ 
      -DENABLE_MKLCPU_BACKEND=False \ 
      -DENABLE_MKLGPU_BACKEND=False \
      -DENABLE_PORTBLAS_BACKEND=True

Note that this is not a tested configuration. This builds oneMKL Interfaces
with the portBLAS backend only, for a generic SYCL device supported by the 
Open DPC++ project.

.. _project_cleanup:

Project Cleanup
###############

Most use-cases involve building the project without the need to clean up the
build directory. However, if you wish to clean up the build directory, you can
delete the ``build`` folder and create a new one. If you wish to clean up the
build files but retain the build configuration, following commands will help you
do so.

.. code-block:: sh

  # If you use "GNU/Unix Makefiles" for building,
  make clean
  
  # If you use "Ninja" for building
  ninja -t clean


.. _build_for_windows_dpcpp:

Building for Windows
####################

The Windows build is similar to the Linux build, albeit that `fewer backends are
supported <https://github.com/oneapi-src/oneMKL?tab=readme-ov-file#windows>`_.
Additionally, the Ninja build system must be used. For example:

.. code-block:: bash

  # Inside <path to onemkl>
  md build && cd build
  cmake .. -G Ninja [-DCMAKE_CXX_COMPILER=<path_to_icx_compiler>\bin\icx] # required only if icx is not found in environment variable PATH
                    [-DCMAKE_C_COMPILER=<path_to_icx_compiler>\bin\icx]   # required only if icx is not found in environment variable PATH
                    [-DMKL_ROOT=<mkl_install_prefix>]                     # required only if environment variable MKLROOT is not set
                    [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]     # required only for testing
                    [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>] # required only for testing
  ninja
  ctest
  cmake --install . --prefix <path_to_install_dir> # required to have full package structure

.. _build_common_problems_dpcpp:

Build FAQ
#########

clangrt builtins lib not found
  Encountered when trying to build oneMKL with some ROCm libraries. There are
  several possible solutions:

  * If building Open DPC++ from source, add ``compiler-rt`` to the external
    projects compile option: ``--llvm-external-projects compiler-rt``.
  * Manually set the variable ``HIP_CXX_COMPILER`` to HIP's toolkit ``clang++``
    path, for instance ``-DHIP_CXX_COMPILER=/opt/rocm/6.1.0/llvm/bin/clang++``.
    oneMKL may fail to link if the clang versions of ``icpx`` and ``rocm`` are
    not compatible.

Could NOT find CBLAS (missing: CBLAS file)
  Encountered when tests are enabled along with the BLAS domain. The tests
  require a reference BLAS implementation, but cannot find one. Either install
  or build a BLAS library and set ``-DREF_BLAS_ROOT``` as described in
  :ref:`building_and_running_tests`. Alternatively, the tests can be disabled by
  setting ``-DBUILD_FUNCTIONAL_TESTS=False``.

error: invalid target ID ''; format is a processor name followed by an optional colon-delimited list of features followed by an enable/disable sign (e.g.,'gfx908:sramecc+:xnack-')
  The HIP_TARGET has not been set. Please see `Building for ROCm`_.

