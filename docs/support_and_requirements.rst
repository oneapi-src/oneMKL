.. _support_and_requirements:

Support and Requirements
========================

Supported Usage Models
----------------------

There are two oneMKL selector layer implementations:


* **Run-time dispatching**\ : The application is linked with the oneMKL
  library and the required backend is loaded at run-time based on device
  vendor (all libraries should be dynamic).

Example of app.cpp with run-time dispatching:

.. code-block:: cpp

   #include "oneapi/mkl.hpp"

   ...
   cpu_dev = sycl::device(sycl::cpu_selector());
   gpu_dev = sycl::device(sycl::gpu_selector());

   sycl::queue cpu_queue(cpu_dev);
   sycl::queue gpu_queue(gpu_dev);

   oneapi::mkl::blas::column_major::gemm(cpu_queue, transA, transB, m, ...);
   oneapi::mkl::blas::column_major::gemm(gpu_queue, transA, transB, m, ...);

How to build an application with run-time dispatching:

.. code-block:: cmd

   $> dpcpp -fsycl –I$ONEMKL/include app.cpp
   $> dpcpp -fsycl app.o –L$ONEMKL/lib –lonemkl


* **Compile-time dispatching**\ : The application uses a templated backend
  selector API where the template parameters specify the required backends and
  third-party libraries and the application is linked with the required oneMKL
  backend wrapper libraries (libraries can be static or dynamic).

Example of app.cpp with compile-time dispatching:

.. code-block:: cpp

   #include "oneapi/mkl.hpp"

   ...
   cpu_dev = sycl::device(sycl::cpu_selector());
   gpu_dev = sycl::device(sycl::gpu_selector());

   sycl::queue cpu_queue(cpu_dev);
   sycl::queue gpu_queue(gpu_dev);

   oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> cpu_selector(cpu_queue);

   oneapi::mkl::blas::column_major::gemm(cpu_selector, transA, transB, m, ...);
   oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::cublas> {gpu_queue}, transA, transB, m, ...);

How to build an application with compile-time dispatching:

.. code-block:: cmd

   $> clang++ -fsycl –I$ONEMKL/include app.cpp
   $> clang++ -fsycl app.o –L$ONEMKL/lib –lonemkl_blas_mklcpu –lonemkl_blas_cublas

Refer to :ref:`Selecting a Compiler <#selecting_a_compiler>` for the choice between ``dpcpp`` and ``clang++`` compilers.

Supported Configurations
------------------------

Supported domains: BLAS, LAPACK, RNG

Linux*
~~~~~~


.. raw:: html

   <table>
       <thead>
           <tr align="center" >
               <th>Domain</th>
               <th>Backend</th>
               <th>Library</th>
               <th>Supported Link Type</th>
           </tr>
       </thead>
       <tbody>
           <tr>
               <td rowspan=4 align="center">BLAS</td>
               <td align="center">x86 CPU</td>
               <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">Intel GPU</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">NVIDIA GPU</td>
               <td align="center">NVIDIA cuBLAS</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">x86 CPU</td>
               <td align="center">NETLIB LAPACK</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td rowspan=2 align="center">LAPACK</td>
               <td align="center">x86 CPU</td>
               <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">Intel GPU</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td rowspan=3 align="center">RNG</td>
               <td align="center">x86 CPU</td>
               <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">Intel GPU</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">NVIDIA GPU</td>
               <td align="center">NVIDIA cuRAND</td>
               <td align="center">Dynamic, Static</td>
           </tr>
       </tbody>
   </table>


Windows*
~~~~~~~~


.. raw:: html

   <table>
       <thead>
           <tr align="center" >
               <th>Domain</th>
               <th>Backend</th>
               <th>Library</th>
               <th>Supported Link Type</th>
           </tr>
       </thead>
       <tbody>
           <tr>
               <td rowspan=3 align="center">BLAS</td>
               <td align="center">x86 CPU</td>
               <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">Intel GPU</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">x86 CPU</td>
               <td align="center">NETLIB LAPACK</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td rowspan=2 align="center">LAPACK</td>
               <td align="center">x86 CPU</td>
               <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">Intel GPU</td>
               <td align="center">Dynamic, Static</td>
           </tr>
           <tr>
               <td align="center">RNG</td>
               <td align="center">x86 CPU</td>
               <td align="center">Intel(R) oneAPI Math Kernel Library</td>
               <td align="center">Dynamic, Static</td>
           </tr>
       </tbody>
   </table>

Hardware Platform Support
-------------------------

* CPU

  * Intel Atom(R) Processors
  * Intel(R) Core(TM) Processor Family
  * Intel(R) Xeon(R) Processor Family

* Accelerators

  * Intel(R) Processor Graphics GEN9
  * NVIDIA(R) TITAN RTX(TM) (Linux* only. cuRAND backend tested also with
    Quadro and A100 GPUs. Not tested with other NVIDIA GPU families and
    products.)

Supported Operating Systems
---------------------------

Linux*
~~~~~~

.. list-table::
   :header-rows: 1

   * - Operating System
     - CPU Host/Target
     - Integrated Graphics from Intel (Intel GPU)
     - NVIDIA GPU
   * - Ubuntu
     - 18.04.3, 19.04
     - 18.04.3, 19.10
     - 18.04.3, 20.04
   * - SUSE Linux Enterprise Server*
     - 15
     - *Not supported*
     - *Not supported*
   * - Red Hat Enterprise Linux\ * (RHEL*\ )
     - 8
     - *Not supported*
     - *Not supported*
   * - Linux* kernel
     - *N/A*
     - 4.11 or higher
     - *N/A*


Windows*
~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Operating System
     - CPU Host/Target
     - Integrated Graphics from Intel (Intel GPU)
   * - Microsoft Windows*
     - 10 (64-bit version only)
     - 10 (64-bit version only)
   * - Microsoft Windows* Server
     - 2016, 2019
     - *Not supported*

Software Requirements
---------------------

**What should I download?**

General:
~~~~~~~~


.. raw:: html

   <table>
       <thead>
           <tr align="center">
               <th>Using Conan</th>
               <th colspan=3> Using CMake Directly </th>
           </tr>
           <tr align="center">
               <th> </th>
               <th> Functional Testing </th>
               <th> Build Only </th>
               <th>Documentation</th>
           </tr>
       </thead>
       <tbody>
           <tr>
               <td colspan=4 align=center> Linux* : GNU* GCC 5.1 or higher
               <br> Windows* : MSVS* 2017 or MSVS* 2019
               (version 16.5 or newer)</td>
           </tr>
           <tr>
               <td rowspan=2> Python 3.6 or higher </td>
               <td colspan=3 align=center> CMake </td>
               <tr>
                   <td colspan=3 align=center> Ninja (optional) </td>
               </tr>
               <tr>
                   <td rowspan=2> Conan C++ package manager </td>
                   <td> GNU* FORTRAN Compiler </td>
                   <td> - </td>
                   <td> Sphinx </td>
               </tr>
               <tr>
                   <td> NETLIB LAPACK </td>
                   <td> - </td>
                   <td> - </td>
               </tr>
           </tr>
       </tbody>
   </table>


Hardware and OS Specific:
~~~~~~~~~~~~~~~~~~~~~~~~~


.. raw:: html

   <table>
       <thead>
           <tr align="center">
               <th>Operating System</th>
               <th>Device</th>
               <th>Package</th>
               <th>Installed by Conan</th>
           </tr>
       </thead>
       <tbody>
           <td rowspan=5> Linux*/Windows* </td>
           <td rowspan=2> x86 CPU </td>
           <td> Intel(R) oneAPI DPC++ Compiler <br> or <br> Intel project for
             LLVM* technology </td>
           <td> No</td>
           <tr>
               <td> Intel(R) oneAPI Math Kernel Library </td>
               <td> Yes </td>
           </tr>
           <td rowspan=3> Intel GPU </td>
           <td> Intel(R) oneAPI DPC++ Compiler </td>
           <td> No </td>
           <tr>
               <td> Intel GPU driver </td>
               <td> No </td>
           </tr>
           <tr>
               <td> Intel(R) oneAPI Math Kernel Library </td>
               <td> Yes </td>
           </tr>
           <td> Linux* only </td>
           <td> NVIDIA GPU </td>
           <td> Intel project for LLVM* technology </td>
           <td> No </td>
       </tbody>
   </table>


*If `Building with Conan <#building-with-conan>`_\ , above packages marked as "No" must be installed manually.*

*If `Building with CMake <#building-with-cmake>`_\ , above packages must be installed manually.*

Notice for Use of Conan Package Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**LEGAL NOTICE: By downloading and using this container or script as applicable (the "Software Package") and the included software or software made available for download, you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software (together, the "Agreements") included in this README file.**

**If the Software Package is installed through a silent install, your download and use of the Software Package indicates your acceptance of the Agreements.**

Product and Version Information:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Product
     - Supported Version
     - Installed by Conan
     - Conan Package Source
     - Package Install Location on Linux*
     - License
   * - Python
     - 3.6 or higher
     - No
     - *N/A*
     - *Pre-installed or Installed by user*
     - `PSF <https://docs.python.org/3.6/license.html>`_
   * - `Conan C++ Package Manager <https://conan.io/downloads.html>`_
     - 1.24 or higher
     - No
     - *N/A*
     - *Installed by user*
     - `MIT <https://github.com/conan-io/conan/blob/develop/LICENSE.md>`_
   * - `CMake <https://cmake.org/download/>`_
     - 3.13 or higher
     - Yes\ :raw-html-m2r:`<br>`\ (3.15 or higher)
     - conan-center
     - ~/.conan/data or $CONAN_USER_HOME/.conan/data
     - `The OSI-approved BSD 3-clause License <https://gitlab.kitware.com/cmake/cmake/raw/master/Copyright.txt>`_
   * - `Ninja <https://ninja-build.org/>`_
     - 1.10.0
     - Yes
     - conan-center
     - ~/.conan/data or $CONAN_USER_HOME/.conan/data
     - `Apache License v2.0 <https://github.com/ninja-build/ninja/blob/master/COPYING>`_
   * - `GNU* FORTRAN Compiler <https://gcc.gnu.org/wiki/GFortran>`_
     - 7.4.0 or higher
     - Yes
     - apt
     - /usr/bin
     - `GNU General Public License, version 3 <https://gcc.gnu.org/onlinedocs/gcc-7.5.0/gfortran/Copying.html>`_
   * - `Intel(R) oneAPI DPC++ Compiler <https://software.intel.com/en-us/oneapi/dpc-compiler>`_
     - latest
     - No
     - *N/A*
     - *Installed by user*
     - `End User License Agreement for the Intel(R) Software Development Products <https://software.intel.com/en-us/license/eula-for-intel-software-development-products>`_
   * - `Intel project for LLVM* technology binary for x86 CPU <https://github.com/intel/llvm/releases>`_
     - Daily builds (experimental) tested with `20200331 <https://github.com/intel/llvm/releases/download/20200331/dpcpp-compiler.tar.gz>`_
     - No
     - *N/A*
     - *Installed by user*
     - `Apache License v2 <https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT>`_
   * - `Intel project for LLVM* technology source for NVIDIA GPU <https://github.com/intel/llvm/releases>`_
     - Daily source releases: tested with `20200421 <https://github.com/intel/llvm/tree/20200421>`_
     - No
     - *N/A*
     - *Installed by user*
     - `Apache License v2 <https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT>`_
   * - `Intel(R) oneAPI Math Kernel Library <https://software.intel.com/en-us/oneapi/onemkl>`_
     - latest
     - Yes
     - apt
     - /opt/intel/inteloneapi/mkl
     - `Intel Simplified Software License <https://software.intel.com/en-us/license/intel-simplified-software-license>`_
   * - `NVIDIA CUDA SDK <https://developer.nvidia.com/cublas>`_
     - 10.2
     - No
     - *N/A*
     - *Installed by user*
     - `End User License Agreement <https://docs.nvidia.com/cuda/eula/index.html>`_
   * - `NETLIB LAPACK <https://www.netlib.org/>`_
     - 3.7.1
     - Yes
     - conan-community
     - ~/.conan/data or $CONAN_USER_HOME/.conan/data
     - `BSD like license <http://www.netlib.org/lapack/LICENSE.txt>`_
   * - `Sphinx <https://www.sphinx-doc.org/en/master/>`_
     - 2.4.4
     - Yes
     - pip
     - ~/.local/bin (or similar user local directory)
     - `BSD License <https://github.com/sphinx-doc/sphinx/blob/3.x/LICENSE>`_


*conan-center: https://api.bintray.com/conan/conan/conan-center*

*conan-community: https://api.bintray.com/conan/conan-community/conan*
