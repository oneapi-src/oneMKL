# oneAPI Math Kernel Library (oneMKL) Interfaces

<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo">

oneMKL Interfaces is an open-source implementation of the oneMKL Data Parallel C++ (DPC++) interface according to the [oneMKL specification](https://spec.oneapi.com/versions/latest/elements/oneMKL/source/index.html). It works with multiple devices (backends) using device-specific libraries underneath.

oneMKL is part of [oneAPI](https://oneapi.io).
<br/><br/>

<table>
    <thead>
        <tr align="center" >
            <th>User Application</th>
            <th>oneMKL Layer</th>
            <th>Third-Party Library</th>
            <th>Hardware Backend</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=12 align="center">oneMKL interface</td>
            <td rowspan=12 align="center">oneMKL selector</td>
            <td align="center"><a href="https://software.intel.com/en-us/oneapi/onemkl">Intel(R) oneAPI Math Kernel Library</a> for x86 CPU</td>
            <td align="center">x86 CPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://software.intel.com/en-us/oneapi/onemkl">Intel(R) oneAPI Math Kernel Library</a> for Intel GPU</td>
            <td align="center">Intel GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/cublas"> NVIDIA cuBLAS</a> for NVIDIA GPU </td>
            <td align="center">NVIDIA GPU</td>
        </tr>
	<tr>
            <td align="center"><a href="https://developer.nvidia.com/cusolver"> NVIDIA cuSOLVER</a> for NVIDIA GPU </td>
            <td align="center">NVIDIA GPU</td>
	</tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/curand"> NVIDIA cuRAND</a> for NVIDIA GPU </td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/cufft"> NVIDIA cuFFT</a> for NVIDIA GPU </td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://ww.netlib.org"> NETLIB LAPACK</a> for x86 CPU </td>
            <td align="center">x86 CPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://rocblas.readthedocs.io/en/rocm-4.5.2/"> AMD rocBLAS</a> for AMD GPU </td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/ROCmSoftwarePlatform/rocSOLVER"> AMD rocSOLVER</a> for AMD GPU </td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/ROCmSoftwarePlatform/rocRAND"> AMD rocRAND</a> for AMD GPU </td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/ROCmSoftwarePlatform/rocFFT">AMD rocFFT</a> for AMD GPU </td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/codeplaysoftware/sycl-blas"> SYCL-BLAS </a></td>
            <td align="center">x86 CPU, Intel GPU, NVIDIA GPU, AMD GPU</td>
        </tr>
    </tbody>
</table>


## Table of Contents

- [Support and Requirements](#support-and-requirements)
- [Documentation](#documentation)
- [FAQs](#faqs)
- [Legal Information](#legal-information)

---

## Support and Requirements

### Supported Usage Models:

There are two oneMKL selector layer implementations:

- **Run-time dispatching**: The application is linked with the oneMKL library and the required backend is loaded at run-time based on device vendor (all libraries should be dynamic).

Example of app.cpp with run-time dispatching:

```cpp
#include "oneapi/mkl.hpp"

...
cpu_dev = sycl::device(sycl::cpu_selector());
gpu_dev = sycl::device(sycl::gpu_selector());

sycl::queue cpu_queue(cpu_dev);
sycl::queue gpu_queue(gpu_dev);

oneapi::mkl::blas::column_major::gemm(cpu_queue, transA, transB, m, ...);
oneapi::mkl::blas::column_major::gemm(gpu_queue, transA, transB, m, ...);
```
How to build an application with run-time dispatching:

if OS is Linux, use icpx compiler. If OS is Windows, use icx compiler.
Linux example:
```cmd
$> icpx -fsycl –I$ONEMKL/include app.cpp
$> icpx -fsycl app.o –L$ONEMKL/lib –lonemkl
```

- **Compile-time dispatching**: The application uses a templated backend selector API where the template parameters specify the required backends and third-party libraries and the application is linked with the required oneMKL backend wrapper libraries (libraries can be static or dynamic).

Example of app.cpp with compile-time dispatching:

```cpp
#include "oneapi/mkl.hpp"

...
cpu_dev = sycl::device(sycl::cpu_selector());
gpu_dev = sycl::device(sycl::gpu_selector());

sycl::queue cpu_queue(cpu_dev);
sycl::queue gpu_queue(gpu_dev);

oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> cpu_selector(cpu_queue);

oneapi::mkl::blas::column_major::gemm(cpu_selector, transA, transB, m, ...);
oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::cublas> {gpu_queue}, transA, transB, m, ...);
```
How to build an application with compile-time dispatching:

```cmd
$> clang++ -fsycl –I$ONEMKL/include app.cpp
$> clang++ -fsycl app.o –L$ONEMKL/lib –lonemkl_blas_mklcpu –lonemkl_blas_cublas
```

*Refer to [Selecting a Compiler](https://oneapi-src.github.io/oneMKL/selecting_a_compiler.html) for the choice between `icpx/icx` and `clang++` compilers.*

### Supported Configurations:

Supported domains: BLAS, LAPACK, RNG, DFT

#### Linux*

<table>
    <thead>
        <tr align="center" >
            <th>Domain</th>
            <th>Backend</th>
            <th>Library</th>
            <th>Supported Link Type</th>
            <th>Supported Compiler</th>		
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=6 align="center">BLAS</td>
            <td align="center">x86 CPU</td>
            <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*, hipSYCL</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuBLAS</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">LLVM*, hipSYCL</td>
        </tr>
        <tr>
            <td align="center">x86 CPU</td>
            <td align="center">NETLIB LAPACK</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*, hipSYCL</td>
        </tr>
	    <tr >
            <td align="center">AMD GPU</td>
            <td align="center">AMD rocBLAS</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">LLVM*, hipSYCL</td>
        </tr>
	    <tr >
            <td align="center">x86 CPU, Intel GPU, NVIDIA GPU, AMD GPU</td>
            <td align="center">SYCL-BLAS</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*</td>
        </tr>
        <tr>
            <td rowspan=4 align="center">LAPACK</td>
            <td align="center">x86 CPU</td>
            <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuSOLVER</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">LLVM*</td>
        </tr>
        <tr>
            <td align="center">AMD GPU</td>
            <td align="center">AMD rocSOLVER</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">LLVM*</td>
        </tr>
        <tr>
            <td rowspan=4 align="center">RNG</td>
            <td align="center">x86 CPU</td>
            <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*, hipSYCL</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuRAND</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">LLVM*, hipSYCL</td>
        </tr>
        <tr>
            <td align="center">AMD GPU</td>
            <td align="center">AMD rocRAND</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">LLVM*, hipSYCL</td>
        </tr>
        <tr>
            <td rowspan=4 align="center">DFT</td>
            <td align="center">Intel GPU</td>
            <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td align="center">x86 CPU</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuFFT</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td align="center">AMD GPU</td>
            <td align="center">AMD rocFFT</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
    </tbody>
</table>

#### Windows*

<table>
    <thead>
        <tr align="center" >
            <th>Domain</th>
            <th>Backend</th>
            <th>Library</th>
            <th>Supported Link Type</th>
            <th>Supported Compiler</th>	
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3 align="center">BLAS</td>
            <td align="center">x86 CPU</td>
            <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td align="center">x86 CPU</td>
            <td align="center">NETLIB LAPACK</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">LAPACK</td>
            <td align="center">x86 CPU</td>
            <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">RNG</td>
            <td align="center">x86 CPU</td>
            <td rowspan=2 align="center">Intel(R) oneAPI Math Kernel Library</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++, LLVM*</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Dynamic, Static</td>
            <td align="center">DPC++</td>
        </tr>
    </tbody>
</table>

\* LLVM - [Intel project for LLVM* technology](https://github.com/intel/llvm) with support for [NVIDIA CUDA](https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda)

---

### Hardware Platform Support

- CPU
    - Intel Atom(R) Processors
    - Intel(R) Core(TM) Processor Family
    - Intel(R) Xeon(R) Processor Family
- Accelerators
    - Intel(R) Processor Graphics GEN9
    - NVIDIA(R) TITAN RTX(TM) (Linux* only. cuRAND backend tested also with Quadro and A100 GPUs. Not tested with other NVIDIA GPU families and products.)
    - AMD(R) GPUs see [here](https://github.com/RadeonOpenCompute/ROCm#hardware-and-software-support) tested on AMD Vega 20 (gfx906)
    
---
### Supported Operating Systems

#### Linux*

Operating System | CPU Host/Target | Integrated Graphics from Intel (Intel GPU) |  NVIDIA GPU
:--- | :--- | :--- | :---
Ubuntu                            | 18.04.3, 19.04 | 18.04.3, 19.10  | 18.04.3, 20.04
SUSE Linux Enterprise Server*     | 15             | *Not supported* | *Not supported*
Red Hat Enterprise Linux* (RHEL*) | 8              | *Not supported* | *Not supported*
Linux* kernel                     | *N/A*          | 4.11 or higher | *N/A*

#### Windows*

Operating System | CPU Host/Target | Integrated Graphics from Intel (Intel GPU)
:--- | :--- | :---
Microsoft Windows* | 10 (64-bit version only) | 10 (64-bit version only)
Microsoft Windows* Server | 2016, 2019 | *Not supported*
---

### Software Requirements

**What should I download?**

#### General:
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
            <td colspan=4 align=center> Linux* : GNU* GCC 5.1 or higher <br> Windows* : MSVS* 2017 or MSVS* 2019 (version 16.5 or newer) </td>
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

#### Hardware and OS Specific:
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
        <td> Intel(R) oneAPI DPC++ Compiler <br> or <br> Intel project for LLVM* technology </td>
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
        <td rowspan=2> Linux* only </td>
        <td> NVIDIA GPU </td>
        <td> Intel project for LLVM* technology <br> or <br> hipSYCL with CUDA backend and dependencies </td>
        <td> No </td>
        <tr>
            <td> AMD GPU </td>
            <td> Intel project for LLVM* technology <br> or <br> hipSYCL with ROCm backend and dependencies </td>
            <td> No </td>
        </tr>
    </tbody>
</table>

*If [Building with Conan](https://oneapi-src.github.io/oneMKL/building_the_project.html#building-with-conan), above packages marked as "No" must be installed manually.*

*If [Building with CMake](https://oneapi-src.github.io/oneMKL/building_the_project.html#building-with-cmake), above packages must be installed manually.*

#### Notice for Use of Conan Package Manager
**LEGAL NOTICE: By downloading and using this container or script as applicable (the "Software Package") and the included software or software made available for download, you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software (together, the "Agreements") included in this README file.**

**If the Software Package is installed through a silent install, your download and use of the
Software Package indicates your acceptance of the Agreements.**

#### Product and Version Information:

Product | Supported Version | Installed by Conan | Conan Package Source | Package Install Location on Linux* | License
:--- | :--- | :--- | :--- | :--- | :---
Python | 3.6 or higher | No | *N/A* | *Pre-installed or Installed by user* | [PSF](https://docs.python.org/3.6/license.html)
[Conan C++ Package Manager](https://conan.io/downloads.html) | 1.24 or higher | No | *N/A* | *Installed by user* | [MIT](https://github.com/conan-io/conan/blob/develop/LICENSE.md)
[CMake](https://cmake.org/download/) | 3.13 or higher | Yes<br>(3.15 or higher) | conan-center | ~/.conan/data or $CONAN_USER_HOME/.conan/data | [The OSI-approved BSD 3-clause License](https://gitlab.kitware.com/cmake/cmake/raw/master/Copyright.txt)
[Ninja](https://ninja-build.org/) | 1.10.0 | Yes | conan-center | ~/.conan/data or $CONAN_USER_HOME/.conan/data | [Apache License v2.0](https://github.com/ninja-build/ninja/blob/master/COPYING)
[GNU* FORTRAN Compiler](https://gcc.gnu.org/wiki/GFortran) | 7.4.0 or higher | Yes | apt | /usr/bin | [GNU General Public License, version 3](https://gcc.gnu.org/onlinedocs/gcc-7.5.0/gfortran/Copying.html)
[Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) | latest | No | *N/A* | *Installed by user* | [End User License Agreement for the Intel(R) Software Development Products](https://software.intel.com/en-us/license/eula-for-intel-software-development-products)
[hipSYCL](https://github.com/illuhad/hipSYCL/) | later than [2cfa530](https://github.com/illuhad/hipSYCL/commit/2cfa5303fd88b8f84e539b5bb6ed41e49c6d6118) | No | *N/A* | *Installed by user* | [BSD-2-Clause License ](https://github.com/illuhad/hipSYCL/blob/develop/LICENSE)
[Intel project for LLVM* technology binary for x86 CPU](https://github.com/intel/llvm/releases) | Daily builds (experimental) tested with [20200331](https://github.com/intel/llvm/releases/download/20200331/dpcpp-compiler.tar.gz) | No | *N/A* | *Installed by user* | [Apache License v2](https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT)
[Intel project for LLVM* technology source for NVIDIA GPU](https://github.com/intel/llvm/releases) | Daily source releases: tested with [20200421](https://github.com/intel/llvm/tree/20200421) | No | *N/A* | *Installed by user* | [Apache License v2](https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT)
[Intel(R) oneAPI Math Kernel Library](https://software.intel.com/en-us/oneapi/onemkl) | latest | Yes | apt | /opt/intel/inteloneapi/mkl | [Intel Simplified Software License](https://software.intel.com/en-us/license/intel-simplified-software-license)
[NVIDIA CUDA SDK](https://developer.nvidia.com/cublas) | 10.2 | No | *N/A* | *Installed by user* |[End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html)
[AMD rocBLAS](https://rocblas.readthedocs.io/en/rocm-4.5.2/) | 4.5 | No | *N/A* | *Installed by user* |[AMD License](https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/LICENSE.md)
[AMD rocRAND](https://github.com/ROCmSoftwarePlatform/rocRAND) | 5.1.0 | No | *N/A* | *Installed by user* |[AMD License](https://github.com/ROCmSoftwarePlatform/rocRAND/blob/develop/LICENSE.txt)
[AMD rocSOLVER](https://github.com/ROCmSoftwarePlatform/rocSOLVER) | 5.0.0 | No | *N/A* | *Installed by user* |[AMD License](https://github.com/ROCmSoftwarePlatform/rocRAND/blob/develop/LICENSE.txt)
[AMD rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT) | rocm-5.4.3 | No | *N/A* | *Installed by user* |[AMD License](https://github.com/ROCmSoftwarePlatform/rocFFT/blob/rocm-5.4.3/LICENSE.md)
[NETLIB LAPACK](https://www.netlib.org/) | 3.7.1 | Yes | conan-community | ~/.conan/data or $CONAN_USER_HOME/.conan/data | [BSD like license](http://www.netlib.org/lapack/LICENSE.txt)
[Sphinx](https://www.sphinx-doc.org/en/master/) | 2.4.4 | Yes | pip | ~/.local/bin (or similar user local directory) | [BSD License](https://github.com/sphinx-doc/sphinx/blob/3.x/LICENSE)
[SYCL-BLAS](https://github.com/codeplaysoftware/sycl-blas) | 0.1 | No | *N/A* | *Installed by user* | [Apache License v2.0](https://github.com/codeplaysoftware/sycl-blas/blob/master/LICENSE)

*conan-center: https://api.bintray.com/conan/conan/conan-center*

*conan-community: https://api.bintray.com/conan/conan-community/conan*

---

## Documentation
- [Contents](https://oneapi-src.github.io/oneMKL/)
- [About](https://oneapi-src.github.io/oneMKL/introduction.html)
- Get Started
  - [Selecting a Compiler](https://oneapi-src.github.io/oneMKL/selecting_a_compiler.html)
  - [Building the Project](https://oneapi-src.github.io/oneMKL/building_the_project.html)
- Developer Reference
  - [oneMKL Defined Datatypes](https://oneapi-src.github.io/oneMKL/onemkl-datatypes.html)
  - [Dense Linear Algebra](https://oneapi-src.github.io/oneMKL/domains/dense_linear_algebra.html)
- [Integrating a Third-Party Library](https://oneapi-src.github.io/oneMKL/create_new_backend.html)

---

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

---

## License

    Distributed under the Apache license 2.0. See [LICENSE](LICENSE) for more
information.

---

## FAQs

### oneMKL

1. What is the difference between the following oneMKL items?
   - The [oneAPI Specification for oneMKL](https://spec.oneapi.com/versions/latest/index.html)
   - The [oneAPI Math Kernel Library (oneMKL) Interfaces](https://github.com/oneapi-src/oneMKL) Project
   - The [Intel(R) oneAPI Math Kernel Library (oneMKL)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) Product

Answer:

- The [oneAPI Specification for oneMKL](https://spec.oneapi.com/versions/latest/index.html) defines the DPC++ interfaces for performance math library functions. The oneMKL specification can evolve faster and more frequently than implementations of the specification.

- The [oneAPI Math Kernel Library (oneMKL) Interfaces](https://github.com/oneapi-src/oneMKL) Project is an open source implementation of the specification. The project goal is to demonstrate how the DPC++ interfaces documented in the oneMKL specification can be implemented for any math library and work for any target hardware. While the implementation provided here may not yet be the full implementation of the specification, the goal is to build it out over time. We encourage the community to contribute to this project and help to extend support to multiple hardware targets and other math libraries.

- The [Intel(R) oneAPI Math Kernel Library (oneMKL)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) product is the Intel product implementation of the specification (with DPC++ interfaces) as well as similar functionality with C and Fortran interfaces, and is provided as part of Intel® oneAPI Base Toolkit. It is highly optimized for Intel CPU and Intel GPU hardware.

### Conan

1. I am behind a proxy. How can Conan download dependencies from external network?
   - `~/.conan/conan.conf` has a `[proxies]` section where you can add the list of proxies. For details refer to [Conan proxy settings](https://docs.conan.io/en/latest/reference/config_files/conan.conf.html#proxies).

2. I get an error while installing packages via APT through Conan.
    ```
    dpkg: warning: failed to open configuration file '~/.dpkg.cfg' for reading: Permission denied
    Setting up intel-oneapi-mkl-devel (2021.1-408.beta07) ...
    E: Sub-process /usr/bin/dpkg returned an error code (1)
    ```
    - Although your user session has permissions to install packages via `sudo apt`, it does not have permissions to update debian package configuration, which throws an error code 1, causing a failure in `conan install` command.
    - The package is most likely installed correctly and can be verified by:
      1. Running the `conan install` command again.
      2. Checking `/opt/intel/inteloneapi` for `mkl` and/or `tbb` directories.

---


#### [Legal information](legal_information.md)
