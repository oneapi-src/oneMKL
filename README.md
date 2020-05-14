# oneAPI Math Kernel Library (oneMKL) Interfaces

## Contents

- [Introduction](#introduction)
- [Support and Requirements](#support-and-requirements)
- [Build Setup](#build-setup)
- [Building with CMake](#building-with-cmake)
- [Project Cleanup](#project-cleanup)
- [Legal Information](#legal-information)

---

## Introduction

oneMKL interfaces are an open-source implementation of the oneMKL Data Parallel C++ (DPC++) interface according to the [oneMKL specification](https://spec.oneapi.com/versions/latest/elements/oneMKL/source/index.html). It works with multiple devices (backends) using device-specific libraries underneath.

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
            <td rowspan=4 align="center">oneMKL interface</td>
            <td rowspan=4 align="center">oneMKL selector</td>
            <td align="center"><a href="https://software.intel.com/en-us/oneapi/onemkl">Intel(R) oneAPI Math Kernel Library</a> for Intel CPU</td>
            <td align="center">Intel CPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://software.intel.com/en-us/oneapi/onemkl">Intel(R) oneAPI Math Kernel Library</a> for Intel GPU</td>
            <td align="center">Intel GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/cublas"> NVIDIA cuBLAS</a> for NVIDIA GPU </td>
            <td align="center">NVIDIA GPU</td>
        </tr>
    </tbody>
</table>

### Supported Usage Models:

There are two oneMKL selector layer implementations:

- **Run-time dispatching**: The application is linked with the oneMKL library and the required backend is loaded at run-time based on device vendor (all libraries should be dynamic).

Example of app.cpp with run-time dispatching:

```cpp
include "onemkl/onemkl.hpp"

...
cpu_dev = cl::sycl::device(cl::sycl::cpu_selector());
gpu_dev = cl::sycl::device(cl::sycl::gpu_selector());

cl::sycl::queue cpu_queue(cpu_dev);
cl::sycl::queue gpu_queue(gpu_dev);

onemkl::blas::gemm(cpu_queue, transA, transB, m, ...);
onemkl::blas::gemm(gpu_queue, transA, transB, m, ...);
```
How to build an application with run-time dispatching:

```cmd
$> clang++ -fsycl –I$ONEMKL/include app.cpp
$> clang++ -fsycl app.o –L$ONEMKL/lib –lonemkl
```

- **Compile-time dispatching**: The application uses a templated API where the template parameters specify the required backends and third-party libraries and the application is linked with the required oneMKL backend wrapper libraries (libraries can be static or dynamic).

Example of app.cpp with compile-time dispatching:

```cpp
include "onemkl/onemkl.hpp"

...
cpu_dev = cl::sycl::device(cl::sycl::cpu_selector());
gpu_dev = cl::sycl::device(cl::sycl::gpu_selector());

cl::sycl::queue cpu_queue(cpu_dev);
cl::sycl::queue gpu_queue(gpu_dev);

onemkl::blas::gemm<intelcpu,intelmkl>(cpu_queue, transA, transB, m, ...);
onemkl::blas::gemm<nvidiagpu,cublas>(gpu_queue, transA, transB, m, ...);
```
How to build an application with run-time dispatching:

```cmd
$> clang++ -fsycl –I$ONEMKL/include app.cpp
$> clang++ -fsycl app.o –L$ONEMKL/lib –lonemkl_blas_mklcpu –lonemkl_blas_cublas
```

### Supported Configurations:

Supported domains: BLAS

#### Linux*

 Backend | Library | Supported Link Type
 :------| :-------| :------------------
 Intel CPU | Intel(R) oneAPI Math Kernel Library | Dynamic, Static
 Intel GPU | Intel(R) oneAPI Math Kernel Library | Dynamic, Static
 NVIDIA GPU | NVIDIA cuBLAS | Dynamic, Static

#### Windows*

 Backend | Library | Supported Link Type
 :------| :-------| :------------------
 Intel CPU | Intel(R) oneAPI Math Kernel Library | Dynamic, Static
 Intel GPU | Intel(R) oneAPI Math Kernel Library | Dynamic, Static
  
---

## Support and Requirements

### Hardware Platform Support

- CPU
    - Intel Atom(R) Processors
    - Intel(R) Core(TM) Processor Family
    - Intel(R) Xeon(R) Processor Family
- Accelerators
    - Intel(R) Processor Graphics GEN9
    - NVIDIA(R) TITAN RTX(TM) (Linux* only. Not tested with other NVIDIA GPU families and products.)
    
---
### Supported Operating Systems

#### Linux*

Operating System | CPU Host/Target | Integrated Graphics from Intel (Intel GPU) |  NVIDIA GPU
:--- | :--- | :--- | :---
Ubuntu                            | 18.04.3, 19.04 | 18.04.3, 19.10  | 18.04.3
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
            <th> Functional Testing </th>
            <th> Build Only </th>
            <th>Documentation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> CMake </td>
            <td> CMake </td>
            <td> CMake </td>
            <tr>
                <td> Ninja (optional) </td>
                <td rowspan=3> Ninja (optional) </td>
                <td rowspan=3> Sphinx </td>
            </tr>
            <tr>
                <td> GNU* FORTRAN Compiler </td>
            </tr>
            <tr>
                <td> NETLIB LAPACK </td>
            </tr>
        </tr>
    </tbody>
</table>


#### Hardware and OS Specific:
<table>
    <thead>
        <tr align="center">
            <th>Operating System</th>
            <th>Hardware</th>
            <th>Using CMake</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3> Linux* </td>
            <td> Any </td>
            <td colspan=2> GNU* GCC 5.1 or higher </td>
            <tr>
                <td rowspan=2> NVIDIA GPU </td>
                <td> Intel project for LLVM* technology </td>
                <tr>
                    <td> NVIDIA CUDA SDK </td>
                </tr>
            </tr>    
        </tr>
        <tr>
            <td rowspan=1> Windows* </td>
            <td> Any </td>
            <td colspan=2> MSVS* 2017 <br> or <br> MSVS* 2019 (version 16.5 or newer) </td>
        </tr>
        <tr>
            <td rowspan=8> Linux*/Windows* </td>
            <tr>
                <td rowspan=2> Intel CPU </td>
                <td> Intel(R) oneAPI DPC++ Compiler <br> or <br> Intel project for LLVM* technology </td>
                <tr>
                    <td> Intel(R) oneAPI Math Kernel Library </td>
                </tr>
            </tr>
            <td rowspan=3> Intel GPU </td>
            <td> Intel(R) oneAPI DPC++ Compiler </td>
            <tr>
                <td> Intel GPU driver </td>
            </tr>
            <tr>
                <td> Intel(R) oneAPI Math Kernel Library </td>
            </tr>
        </tr>    
    </tbody>
</table>

#### Product and Version Information:

Product | Supported Version | License
:--- | :--- | :---
Python | 3.6 or higher | [PSF](https://docs.python.org/3.6/license.html)
[CMake](https://cmake.org/download/) | 3.13 or higher | [The OSI-approved BSD 3-clause License](https://gitlab.kitware.com/cmake/cmake/raw/master/Copyright.txt)
[Ninja](https://ninja-build.org/) | 1.9.0 | [Apache License v2.0](https://github.com/ninja-build/ninja/blob/master/COPYING)
[GNU* FORTRAN Compiler](https://gcc.gnu.org/wiki/GFortran) | 7.4.0 or higher | [GNU General Public License, version 3](https://gcc.gnu.org/onlinedocs/gcc-7.5.0/gfortran/Copying.html)
[Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) | 2021.1-beta05 | [End User License Agreement for the Intel(R) Software Development Products](https://software.intel.com/en-us/license/eula-for-intel-software-development-products)
[Intel project for LLVM* technology binary for Intel CPU](https://github.com/intel/llvm/releases) | Daily builds (experimental) tested with [20200331](https://github.com/intel/llvm/releases/download/20200331/dpcpp-compiler.tar.gz) | [Apache License v2](https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT)
[Intel project for LLVM* technology source for NVIDIA GPU](https://github.com/intel/llvm/releases) | Daily source releases: tested with [20200421](https://github.com/intel/llvm/tree/20200421) | [Apache License v2](https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT)
[Intel(R) oneAPI Math Kernel Library](https://software.intel.com/en-us/oneapi/onemkl) | 2021.1-beta05 | [Intel Simplified Software License](https://software.intel.com/en-us/license/intel-simplified-software-license)
[NVIDIA CUDA SDK](https://developer.nvidia.com/cublas) | 10.2 | [End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html)
[NETLIB LAPACK](https://github.com/Reference-LAPACK/lapack) | 3.7.1 | [BSD like license](http://www.netlib.org/lapack/LICENSE.txt)
[Sphinx](https://www.sphinx-doc.org/en/master/) | 2.4.4 | [BSD License](https://github.com/sphinx-doc/sphinx/blob/3.x/LICENSE)

---

## Build Setup

1. Install Intel(R) oneAPI DPC++ Compiler (select variant as per requirement).

2. Clone this project to `<path to onemkl>`, where `<path to onemkl>` is the root directory of this repository.

3. [Build with CMake](#building-with-cmake).

---

## Building with CMake

1. Make sure you have completed [Build Setup](#build-setup). 

2. Build and install all required [dependencies](#software-requirements). 

Then:

- On Linux*
```bash
# Inside <path to onemkl>
mkdir build && cd build
export CXX=<path_to_dpcpp_compiler>/bin/dpcpp;
cmake .. [-DMKL_ROOT=<mkl_install_prefix>] \               # required only if enviroment variable MKLROOT is not set
         [-DREF_BLAS_ROOT=<reference_blas_install_prefix>] # required only for testing
cmake --build .
ctest
cmake --install . --prefix <path_to_install_dir>
```
- On Windows*
```bash
# Inside <path to onemkl>
md build && cd build
cmake .. -G Ninja  -DCMAKE_TOOLCHAIN_FILE="..\cmake\toolchain\intel_clang-cl-toolchain.cmake"  
                  [-DMKL_ROOT=<mkl_install_prefix>] \                   # required only if enviroment variable MKLROOT is not set
                  [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]     # required only for testing

ninja 
ctest
cmake --install . --prefix <path_to_install_dir>
```

### Build Options
You can specify build options using `-D<cmake_option>=<value>`. The following table provides the list of options supported by CMake.

CMake Option | Supported Values | Default Value
:----------- | :--------------- | :---
BUILD_SHARED_LIBS        | True, False         | True
ENABLE_CUBLAS_BACKEND    | True, False         | False
ENABLE_MKLCPU_BACKEND    | True, False         | True
ENABLE_MKLGPU_BACKEND    | True, False         | True
ENABLE_MKLCPU_THREAD_TBB | True, False         | True
BUILD_FUNCTIONAL_TESTS   | True, False         | True
BUILD_DOC                | True, False         | False

---

## Project Cleanup

Most use-cases involve building the project without the need to cleanup the build directory. However, if you wish to cleanup the build directory, you can delete the `build` folder and create a new one. If you wish to cleanup the build files but retain the build configuration, following commands will help you do so.

```sh
# If you use "GNU/Unix Makefiles" for building,
make clean

# If you use "Ninja" for building
ninja -t clean
```

---

## [Legal information](legal_information.md)
