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
   You can :ref:`Build with Conan <building_with_conan>` to automate the
   process of getting dependencies or you can download and install the
   required dependencies manually and
   :ref:`Build with CMake <building_with_cmake>` directly.

.. note::
  Conan package manager automates the process of getting required packages
  so that you do not have to go to different web location and follow different
  instructions to install them.

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

.. _building_with_conan:

Building with Conan
###################

** This method currently works on Linux* only **

** Make sure you have completed :ref:`Build Setup <build_setup>`. **

.. note::
  To understand how dependencies are resolved, refer to "Product and Version
  Information" under
  `Support and Requirements <https://github.com/oneapi-src/oneMKL#support-and-requirements>`_.
  For details about Conan package manager, refer to the
  `Conan Documentation <https://docs.conan.io/en/latest/>`_.

Getting Conan
^^^^^^^^^^^^^

Conan can be `installed <https://docs.conan.io/en/latest/installation.html>`_ from pip:

.. code-block:: bash

   pip3 install conan

Setting up Conan
^^^^^^^^^^^^^^^^

Conan Default Directory
~~~~~~~~~~~~~~~~~~~~~~~

Conan stores all files and data in ``~/.conan``. If you are fine with this
behavior, you can skip to the :ref:`Conan Profiles <conan-profiles>` section.

To change this behavior, set the environment variable ``CONAN_USER_HOME`` to a
path of your choice. A ``.conan/`` directory will be created in this path and
future Conan commands will use this directory to find configuration files and
download dependent packages. Packages will be downloaded into
``$CONAN_USER_HOME/data``. To change the ``"/data"`` part of this directory,
refer to the ``[storage]`` section of ``conan.conf`` file.

To make this setting persistent across terminal sessions, you can add the
line below to your ``~/.bashrc`` or custom runscript. Refer to the
`Conan Documentation <https://docs.conan.io/en/latest/reference/env_vars.html#conan-user-home>`_
for more details.

.. code-block:: sh

   export CONAN_USER_HOME=/usr/local/my_workspace/conan_cache

.. _conan-profiles:

Conan Profiles
~~~~~~~~~~~~~~

Profiles are a way for Conan to determine a basic environment to use for
building a project. This project ships with profiles for:


* Intel(R) oneAPI DPC++ Compiler for x86 CPU and Intel GPU backend: ``inteldpcpp_lnx``


#. Open the profile you wish to use from ``<path to onemkl>/conan/profiles/``
   and set ``COMPILER_PREFIX`` to the path to the root folder of compiler.
   The root folder is the one that contains the ``bin`` and ``lib``
   directories. For example, Intel(R) oneAPI DPC++ Compiler root folder for
   default installation on Linux is
   ``/opt/intel/inteloneapi/compiler/<version>/linux``. The user can define a
   custom path for installing the compiler.

.. code-block:: ini

   COMPILER_PREFIX=<path to Intel(R) oneAPI DPC++ Compiler>


#. 
   You can customize the ``[env]`` section of the profile based on individual
   requirements.

#. 
   Install configurations for this project:

   .. code-block:: sh

      # Inside <path to onemkl>
      $ conan config install conan/

   This command installs all contents of ``<path to onemkl>/conan/``\ , most
   importantly profiles, to conan default directory.

.. note::
  If you change the profile, you must re-run the above command before you can
  use the new profile.

Building
^^^^^^^^

#. 
   Out-of-source build

   .. code-block:: bash

      # Inside <path to onemkl>
      mkdir build && cd build

#. 
   If you choose to build backends with the Intel(R) oneAPI
   Math Kernel Library, install the GPG key as mentioned here:
   https://software.intel.com/en-us/articles/oneapi-repo-instructions#aptpkg

#. 
   Install dependencies

   .. code-block:: sh

      conan install .. --profile <profile_name> --build missing [-o <option1>=<value1>] [-o <option2>=<value2>]

   The ``conan install`` command downloads and installs all requirements for
   the oneMKL DPC++ Interfaces project as defined in
   ``<path to onemkl>/conanfile.py`` based on the options passed. It also
   creates ``conanbuildinfo.cmake`` file that contains information about all
   dependencies and their directories. This file is used in top-level
   ``CMakeLists.txt``.

``-pr | --profile <profile_name>``
Defines a profile for Conan to use for building the project.

``-b | --build <package_name|missing>``
Tells Conan to build or re-build a specific package. If ``missing`` is passed
as a value, all missing packages are built. This option is recommended when
you build the project for the first time, because it caches required packages.
You can skip this option for later use of this command.


#. Build Project
   .. code-block:: sh

      conan build .. [--configure] [--build] [--test]  # Default is all

The ``conan build`` command executes the ``build()`` procedure from
``<path to onemkl>/conanfile.py``. Since this project uses ``CMake``\ , you
can choose to ``configure``\ , ``build``\ , ``test`` individually or perform
all steps by passing no optional arguments.


#. Optionally, you can also install the package. Similar to ``cmake --install . --prefix <install_dir>``.

.. code-block:: sh

   conan package .. --build-folder . --install-folder <install_dir>

``-bf | --build-folder``
Tells Conan where to find the built project.

``-if | --install-folder``
Tells Conan where to install the package. It is similar to specifying ``CMAKE_INSTALL_PREFIX``

.. note::
   For a detailed list of commands and options, refer to the
   `Conan Command Reference <https://docs.conan.io/en/latest/reference/commands.html>`_.

Conan Build Options
^^^^^^^^^^^^^^^^^^^

Backend-Related Options
~~~~~~~~~~~~~~~~~~~~~~~

The following ``options`` are available to pass on ``conan install`` when
building the oneMKL library:


* ``build_shared_libs=[True | False]``. Setting it to ``True`` enables the building of dynamic libraries. The default value is ``True``.
* ``target_domains=[<list of values>]``. Setting it to ``blas`` or any other list of domain(s), enables building of those specific domain(s) only. If not defined, the default value is all supported domains.
* ``enable_mklcpu_backend=[True | False]``. Setting it to ``True`` enables the building of oneMKL mklcpu backend. The default value is ``True``.
* ``enable_mklgpu_backend=[True | False]``. Setting it to ``True`` enables the building of oneMKL mklgpu backend. The default value is ``True``.
* ``enable_mklcpu_thread_tbb=[True | False]``. Setting it to ``True`` enables oneMKL on CPU with TBB threading instead of sequential. The default value is ``True``.

Testing-Related Options
~~~~~~~~~~~~~~~~~~~~~~~

* ``build_functional_tests=[True | False]``. Setting it to ``True`` enables
  the building of functional tests. The default value is ``True``.

Example-Related Options
~~~~~~~~~~~~~~~~~~~~~~~

* ``build_examples=[True | False]``. Setting it to ``True`` enables
  the building of examples. The default value is ``True``. Compile_time_dispatching examples will always be built if this value is set to true. Run_time_dispatching examples will be build if both this value and  ``build_shared_libs`` is set to true
  
Documentation
~~~~~~~~~~~~~

* ``build_doc=[True | False]``. Setting it to ``True`` enables the building of rst files to generate HTML files for updated documentation. The default value is ``False``.

.. note::
  For a mapping between Conan and CMake options, refer to
  :ref:`Building with CMake <building_with_cmake>`.

Example
^^^^^^^

Build oneMKL as a static library for oneMKL cpu and gpu backend:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

   # Inside <path to onemkl>
   mkdir build && cd build
   conan install .. --build missing --profile inteldpcpp_lnx -o build_shared_libs=False
   conan build ..

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
   cmake --install . --prefix <path_to_install_dir>

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
            -DTARGET_DOMAINS=blas                               # hipSYCL supports BLAS and RNG domains
            -DHIPSYCL_TARGETS=omp\;hip:gfx906                   # Specify the targetted device architectures 
            -DONEMKL_SYCL_IMPLEMENTATION=hipSYCL                # Use the hipSYCL cmake integration
            [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]   # required only for testing
   cmake --build .
   ctest
   cmake --install . --prefix <path_to_install_dir>

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
   export SYCL_DEVICE_FILTER=HIP
   ctest
   cmake --install . --prefix <path_to_install_dir>

To build with the rocRAND backend instead simply replace:

.. code-block:: bash

   -DENABLE_ROCBLAS_BACKEND=True   \
   -DTARGET_DOMAINS=blas

With:

.. code-block:: bash

   -DENABLE_ROCRAND_BACKEND=True   \
   -DTARGET_DOMAINS=rng

To build with the rocSOLVER backend instead simply replace:

.. code-block:: bash\

   -DENABLE_ROCBLAS_BACKEND=True   \
   -DTARGET_DOMAINS=blas
With:

.. code-block:: bash

   -DENABLE_ROCSOLVER_BACKEND=True   \
   -DTARGET_DOMAINS=lapack

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
   * - build_examples
     - BUILD_EXAMPLES
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


.. note::
  When building with clang++ for AMD backends, you must additionally set
  ``SYCL_DEVICE_FILTER`` to ``HIP`` and provide ``-DHIP_TARGETS`` according to
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
you do so. They apply to both ``Conan`` and ``CMake`` methods of building
this project.

.. code-block:: sh

   # If you use "GNU/Unix Makefiles" for building,
   make clean

   # If you use "Ninja" for building
   ninja -t clean
