# oneAPI Math Kernel Library (oneMKL) Interfaces

<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo">

oneMKL interfaces are an open-source implementation of the oneMKL Data Parallel C++ (DPC++) interface according to the [oneMKL specification](https://spec.oneapi.com/versions/latest/elements/oneMKL/source/index.html). It works with multiple devices (backends) using device-specific libraries underneath.

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
            <td rowspan=6 align="center">oneMKL interface</td>
            <td rowspan=6 align="center">oneMKL selector</td>
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
            <td align="center"><a href="https://ww.netlib.org"> NETLIB LAPACK</a> for x86 CPU </td>
            <td align="center">x86 CPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://rocblas.readthedocs.io/en/rocm-4.5.2/"> AMD rocBLAS</a> for AMD GPU </td>
            <td align="center">AMD GPU</td>
        </tr>
    </tbody>
</table>


## Table of Contents

- [Support and Requirements](#support-and-requirements)
- [Selection of Compilers](#selection-of-compilers)
- [Build Setup](#build-setup)
- [Building with Conan](#building-with-conan)
- [Building with CMake](#building-with-cmake)
- [Project Cleanup](#project-cleanup)
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

```cmd
$> dpcpp -fsycl –I$ONEMKL/include app.cpp
$> dpcpp -fsycl app.o –L$ONEMKL/lib –lonemkl
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

*Refer to [Selection of Compilers](#selection-of-compilers) for the choice between `dpcpp` and `clang++` compilers.*

### Supported Configurations with DPC++ compiler:

Supported domains: BLAS, LAPACK, RNG

#### Linux*

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
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuSOLVER</td>
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

#### Windows*

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

### Supported Configurations with hipSYCL:

Supported domains: BLAS

#### Linux*

<table>
<thead>
<tr  align="center">
<th >Domain</th>
<th >Backend</th>
<th >Library</th>
<th >Supported Link Type</th>
</tr>
</thead>
<tbody>
<tr >
<td  rowspan="4" align="center">BLAS</td>
<td  align="center">x86 CPU</td>
<td  align="center">Intel(R) oneAPI Math Kernel Library</td>
<td  align="center">Dynamic, Static</td>
</tr>
<tr >
<td align="center">AMD GPU</td>
<td align="center">AMD rocBLAS </td>
<td align="center">Dynamic, Static</td>
</tr>
<tr >
<td  align="center">NVIDIA GPU</td>
<td  align="center">NVIDIA cuBLAS</td>
<td  align="center">Dynamic, Static</td>
</tr>
<tr >
<td  align="center">x86 CPU</td>
<td  align="center">NETLIB LAPACK</td>
<td  align="center">Dynamic, Static</td>
</tr>
</tbody>
</table>


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
        <td> Intel project for LLVM* technology </td>
        <td> No </td>
        <tr>
            <td>AMD GPU</td>
            <td>hipSYCL with ROCm backend and dependencies </td>
            <td>No</td>
        </tr>
    </tbody>
</table>

*If [Building with Conan](#building-with-conan), above packages marked as "No" must be installed manually.*

*If [Building with CMake](#building-with-cmake), above packages must be installed manually.*

#### Notice for Use of Conan Package Manager
**LEGAL NOTICE: By downloading and using this container or script as applicable (the “Software Package”) and the included software or software made available for download, you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software (together, the “Agreements”) included in this README file.**

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
[NETLIB LAPACK](https://www.netlib.org/) | 3.7.1 | Yes | conan-community | ~/.conan/data or $CONAN_USER_HOME/.conan/data | [BSD like license](http://www.netlib.org/lapack/LICENSE.txt)
[Sphinx](https://www.sphinx-doc.org/en/master/) | 2.4.4 | Yes | pip | ~/.local/bin (or similar user local directory) | [BSD License](https://github.com/sphinx-doc/sphinx/blob/3.x/LICENSE)

*conan-center: https://api.bintray.com/conan/conan/conan-center*

*conan-community: https://api.bintray.com/conan/conan-community/conan*

---

## Selection of Compilers

A compiler needs to be chosen according to the required backend of your application.

- If your application requires Intel GPU, use [Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) `dpcpp`.
- If your application requires NVIDIA GPU, use the latest release of `clang++` from [Intel project for LLVM* technology](https://github.com/intel/llvm/releases).
- If your application requires AMD GPU, use `hipSYCL` from the [hipSYCL repository](https://github.com/illuhad/hipSYCL)
- If no Intel GPU, NVIDIA GPU or AMD GPU is required, you can use either [Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) `dpcpp`, `clang++` or `hipSYCL` on Linux and `clang-cl` on Windows from [Intel project for LLVM* technology](https://github.com/intel/llvm/releases).

---

## Build Setup

1. Install Intel(R) oneAPI DPC++ Compiler (select variant as per requirement).

2. Clone this project to `<path to onemkl>`, where `<path to onemkl>` is the root directory of this repository.

3. You can [Build with Conan](#building-with-conan) to automate the process of getting dependencies or you can download and install the required dependencies manually and [Build with CMake](#building-with-cmake) directly.

*Note: Conan package manager automates the process of getting required packages, so that you do not have to go to different web location and follow different instructions to install them.*

---
## Build Setup with hipSYCL

1. Make sure that the dependencies of hipSYCL are fulfilled. For detailed description see the [hipSYCL installation readme](https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md#software-dependencies)

2. Install hipSYCL with the prefered backends enabled. HipSYCL supports various backends. Support can be customized for the target system in compile time by setting the appropriate configuration flags: see the [hipSYCL documentation](https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md) for instructions.

3. Install AMD rocBLAS see instructions [here](https://rocblas.readthedocs.io/en/master/install.html)

4. Clone this project to `<path to onemkl>`, where `<path to onemkl>` is the root directory of this repository.

5. Download and install the required dependencies manually and [Build with CMake](#building-with-cmake).

---

## Building with Conan

** This method currently works on Linux* only **

** Make sure you have completed [Build Setup](#build-setup). **

*Note: To understand how dependencies are resolved, refer to the [Product and Version Information](#product-and-version-information) section. For details about Conan package manager, refer to [Conan Documentation](https://docs.conan.io/en/latest/).*

### Getting Conan
Conan can be [installed](https://docs.conan.io/en/latest/installation.html) from pip:
```bash
pip3 install conan
```

### Setting up Conan

#### Conan Default Directory

Conan stores all files and data in `~/.conan`. If you are fine with this behavior, you can skip to [Conan Profiles](#conan-profiles) section.

To change this behavior, set the environment variable `CONAN_USER_HOME` to a path of your choice. A `.conan/` directory will be created in this path and future Conan commands will use this directory to find configuration files and download dependent packages. Packages will be downloaded into `$CONAN_USER_HOME/data`. To change the `"/data"` part of this directory, refer to the `[storage]` section of `conan.conf` file.

To make this setting persistent across terminal sessions, you can add below line to your `~/.bashrc` or custom runscript. Refer to [Conan Documentation](https://docs.conan.io/en/latest/reference/env_vars.html#conan-user-home) for more details.

```sh
export CONAN_USER_HOME=/usr/local/my_workspace/conan_cache
```

#### Conan Profiles

Profiles are a way for Conan to determine a basic environment to use for building a project. This project ships with profiles for:

- Intel(R) oneAPI DPC++ Compiler for x86 CPU and Intel GPU backend: `inteldpcpp_lnx`

1. Open the profile you wish to use from `<path to onemkl>/conan/profiles/` and set `COMPILER_PREFIX` to the path to the root folder of compiler. The root folder is the one that contains the `bin` and `lib` directories. For example, Intel(R) oneAPI DPC++ Compiler root folder for default installation on Linux is `/opt/intel/inteloneapi/compiler/<version>/linux`. User can define custom path for installing the compiler.

```ini
COMPILER_PREFIX=<path to Intel(R) oneAPI DPC++ Compiler>
```

2. You can customize the `[env]` section of the profile based on individual requirements.

3. Install configurations for this project:
```sh
# Inside <path to onemkl>
$ conan config install conan/
```
This command installs all contents of `<path to onemkl>/conan/`, most importantly profiles, to conan default directory.

*Note: If you change the profile, you must re-run the above command before you can use the new profile.*

### Building

1. Out-of-source build
```bash
# Inside <path to onemkl>
mkdir build && cd build
```

2. If you choose to build backends with the Intel(R) oneAPI Math Kernel Library, install the GPG key as mentioned here, https://software.intel.com/en-us/articles/oneapi-repo-instructions#aptpkg

3. Install dependencies
```sh
conan install .. --profile <profile_name> --build missing [-o <option1>=<value1>] [-o <option2>=<value2>]
```
The `conan install` command downloads and installs all requirements for the oneMKL DPC++ Interfaces project as defined in `<path to onemkl>/conanfile.py` based on the options passed. It also creates `conanbuildinfo.cmake` file that contains information about all dependencies and their directories. This file is used in top-level `CMakeLists.txt`.

`-pr | --profile <profile_name>`
Defines a profile for Conan to use for building the project.

`-b | --build <package_name|missing>`
Tells Conan to build or re-build a specific package. If `missing` is passed as a value, all missing packages are built. This option is recommended when you build the project for the first time, because it caches required packages. You can skip this option for later use of this command.

4. Build Project
```sh
conan build .. [--configure] [--build] [--test]  # Default is all
```

The `conan build` command executes the `build()` procedure from `<path to onemkl>/conanfile.py`. Since this project uses `CMake`, you can choose to `configure`, `build`, `test` individually or perform all steps by passing no optional arguments.

5. Optionally, you can also install the package. Similar to `cmake --install . --prefix <install_dir>`.

```sh
conan package .. --build-folder . --install-folder <install_dir>
```

`-bf | --build-folder`
Tells Conan where to find the built project.

`-if | --install-folder`
Tells Conan where to install the package. It is similar to specifying `CMAKE_INSTALL_PREFIX`

*Note: For a detailed list of commands and options, refer to the [Conan Command Reference](https://docs.conan.io/en/latest/reference/commands.html).*

### Conan Build Options

#### Backend-related Options

The following `options` are available to pass on `conan install` when building the oneMKL library:

- `build_shared_libs=[True | False]`. Setting it to `True` enables the building of dynamic libraries. The default value is `True`.
- `target_domains=[<list of values>]`. Setting it to `blas` or any other list of domain(s), enables building of those specific domain(s) only. If not defined, the default value is all supported domains.
- `enable_mklcpu_backend=[True | False]`. Setting it to `True` enables the building of oneMKL mklcpu backend. The default value is `True`.
- `enable_mklgpu_backend=[True | False]`. Setting it to `True` enables the building of oneMKL mklgpu backend. The default value is `True`.
- `enable_mklcpu_thread_tbb=[True | False]`. Setting it to `True` enables oneMKL on CPU with TBB threading instead of sequential. The default value is `True`.

#### Testing-related Options
- `build_functional_tests=[True | False]`. Setting it to `True` enables the building of functional tests. The default value is `True`.

#### Documentation
- `build_doc=[True | False]`. Setting it to `True` enables the building of rst files to generate HTML files for updated documentation. The default value is `False`.

*Note: For a mapping between Conan and CMake options, refer to [build options](#build-options) under the CMake section.*

### Example
#### Build oneMKL as a static library for oneMKL cpu and gpu backend:
```sh
# Inside <path to onemkl>
mkdir build && cd build
conan install .. --build missing --profile inteldpcpp_lnx -o build_shared_libs=False
conan build ..
```

---

## Building with CMake

1. Make sure you have completed [Build Setup](#build-setup). 

2. Build and install all required [dependencies](#software-requirements). 

### Building for oneMKL

- On Linux*
```bash
# Inside <path to onemkl>
mkdir build && cd build
cmake .. [-DCMAKE_CXX_COMPILER=<path_to_dpcpp_compiler>/bin/dpcpp] \  # required only if dpcpp is not found in environment variable PATH
         [-DCMAKE_C_COMPILER=<path_to_icx_compiler>/bin/icx]       \  # required only if icx is not found in environment variable PATH
         [-DMKL_ROOT=<mkl_install_prefix>] \                          # required only if environment variable MKLROOT is not set
         [-DREF_BLAS_ROOT=<reference_blas_install_prefix>] \          # required only for testing
         [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>]        # required only for testing
cmake --build .
ctest
cmake --install . --prefix <path_to_install_dir>
```
- On Windows*
```bash
# Inside <path to onemkl>
md build && cd build
cmake .. -G Ninja [-DCMAKE_CXX_COMPILER=<path_to_dpcpp_compiler>\bin\dpcpp] \  # required only if dpcpp is not found in environment variable PATH
                  [-DCMAKE_C_COMPILER=<path_to_icx_compiler>\bin\icx]       \  # required only if icx is not found in environment variable PATH
                  [-DMKL_ROOT=<mkl_install_prefix>] \                          # required only if environment variable MKLROOT is not set
                  [-DREF_BLAS_ROOT=<reference_blas_install_prefix>] \          # required only for testing
                  [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>]        # required only for testing
ninja 
ctest
cmake --install . --prefix <path_to_install_dir>
```

### Building for CUDA

- On Linux*

With the cuBLAS backend:

```bash
# Inside <path to onemkl>
mkdir build && cd build
cmake .. [-DCMAKE_CXX_COMPILER=<path_to_clang++_compiler>/bin/clang++] \  # required only if clang++ is not found in environment variable PATH
         [-DCMAKE_C_COMPILER=<path_to_clang_compiler>/bin/clang]       \  # required only if clang is not found in environment variable PATH
         -DENABLE_CUBLAS_BACKEND=True  \
         -DENABLE_MKLCPU_BACKEND=False \                                  # disable Intel MKL CPU backend
         -DENABLE_MKLGPU_BACKEND=False \                                  # disable Intel MKL GPU backend
         [-DREF_BLAS_ROOT=<reference_blas_install_prefix>] \              # required only for testing
cmake --build .
ctest
cmake --install . --prefix <path_to_install_dir>
```

To build with the cuSOLVER or cuRAND backend instead simply replace:
```bash
-DENABLE_CUBLAS_BACKEND=True   \
```

With:
```bash
-DENABLE_CUSOLVER_BACKEND=True   \
```

or

```bash
-DENABLE_CURAND_BACKEND=True   \
```

#### Building for ROCm (with hipSYCL)

With the AMD rocBLAS backend:

- On Linux*

```bash
# Inside <path to onemkl>
mkdir build && cd build
cmake .. -DENABLE_CUBLAS_BACKEND=False                     \
         -DENABLE_MKLCPU_BACKEND=False/True                \   # hipSYCL supports MKLCPU backend     
         -DENABLE_NETLIB_BACKEND=False/True                \   # hipSYCL supports NETLIB backend
         -DENABLE_MKLGPU_BACKEND=False                     \   # disable Intel MKL GPU backend
         -DENABLE_ROCBLAS_BACKEND=True                     \
         -DTARGET_DOMAINS=blas                             \   # hipSYCL only supports the BLAS domain
         -DHIPSYCL_TARGETS=omp\;hip:gfx906                 \   # Specify the targetted device architectures 
         -DONEMKL_SYCL_IMPLEMENTATION=hipSYCL              \   # Use the hipSYCL cmake integration
         [-DREF_BLAS_ROOT=<reference_blas_install_prefix>] \   # required only for testing
cmake --build .
ctest
cmake --install . --prefix <path_to_install_dir>
```

**AMD GPU device architectures**  

The device architecture can be retrieved via the `rocminfo` tool. The architecture will be displayed in the `Name:` row.

A few often used architectures are listed below:
| architecture | AMD GPU name |
| ----         | ----         |
| gfx906       | AMD Radeon Instinct(TM) MI50/60 Accelerator <br> AMD Radeon(TM) (Pro) VII Graphics Card|
| gfx908       | AMD Instinct(TM) MI 100 Accelerator |
| gfx900       | Radeon Instinct(TM) MI 25 Accelerator<br> Radeon(TM) RX Vega 64/56 Graphics|


### Build Options
When building oneMKL the SYCL implementation can be determined, by setting the `ONEMKL_SYCL_IMPLEMENTATION` option. Possible values are 
- `dpc++` (default) for the [Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) and for the `clang++` from [Intel project for LLVM* technology](https://github.com/intel/llvm/releases) compilers. 
- `hipsycl` for the [hipSYCL](https://github.com/illuhad/hipSYCL) SYCL implementation

In the following tables, the supported options for each type of SYCL implementation are listed.

All options specified in the Conan section are available to CMake. You can specify these options using `-D<cmake_option>=<value>`.

The following table provides a detailed mapping of options between Conan and CMake.

**ONEMKL_SYCL_IMPLEMENTATION=dpc++(default)**
Conan Option | CMake Option | Supported Values | Default Value
 :---------- | :----------- | :--------------- | :---
build_shared_libs        | BUILD_SHARED_LIBS        | True, False         | True
enable_mklcpu_backend    | ENABLE_MKLCPU_BACKEND    | True, False         | True
enable_mklgpu_backend    | ENABLE_MKLGPU_BACKEND    | True, False         | True
*Not Supported*          | ENABLE_CUBLAS_BACKEND    | True, False         | False
*Not Supported*          | ENABLE_CUSOLVER_BACKEND  | True, False         | False
*Not Supported*          | ENABLE_CURAND_BACKEND    | True, False         | False
*Not Supported*          | ENABLE_NETLIB_BACKEND    | True, False         | False
*Not Supported*          | ENABLE_ROCBLAS_BACKEND   | False               | False
enable_mklcpu_thread_tbb | ENABLE_MKLCPU_THREAD_TBB | True, False         | True
build_functional_tests   | BUILD_FUNCTIONAL_TESTS   | True, False         | True
build_doc                | BUILD_DOC                | True, False         | False
target_domains (list)    | TARGET_DOMAINS (list)    | blas, lapack, rng   | All domains

**ONEMKL_SYCL_IMPLEMENTATION=hipsycl**
Conan Option | CMake Option | Supported Values | Default Value
 :---------- | :----------- | :--------------- | :---
build_shared_libs        | BUILD_SHARED_LIBS        | True, False         | True
enable_mklcpu_backend    | ENABLE_MKLCPU_BACKEND    | True, False         | True
enable_mklgpu_backend    | ENABLE_MKLGPU_BACKEND    | False               | False
*Not Supported*          | ENABLE_CUBLAS_BACKEND    | False               | False
*Not Supported*          | ENABLE_CURAND_BACKEND    | False               | False
*Not Supported*          | ENABLE_NETLIB_BACKEND    | True, False         | False
*Not Supported*          | ENABLE_ROCBLAS_BACKEND   | True, False         | False
enable_mklcpu_thread_tbb | ENABLE_MKLCPU_THREAD_TBB | True, False         | True
build_functional_tests   | BUILD_FUNCTIONAL_TESTS   | True, False         | True
build_doc                | BUILD_DOC                | True, False         | False
target_domains (list)    | TARGET_DOMAINS (list)    | blas                | All domains
N/A                      | HIPSYCL_TARGETS          | depends on target device | none

*Note: `build_functional_tests` and related CMake option affects all domains at a global scope.*

*Note: When building with hipSYCL, `-DHIPSYCL_TARGETS` additionally needs to be provided according to the targeted hardware. For the options, see the tables in the hipSYCL specific sections*
---

## Project Cleanup

Most use-cases involve building the project without the need to cleanup the build directory. However, if you wish to cleanup the build directory, you can delete the `build` folder and create a new one. If you wish to cleanup the build files but retain the build configuration, following commands will help you do so. They apply to both `Conan` and `CMake` methods of building this project.

```sh
# If you use "GNU/Unix Makefiles" for building,
make clean

# If you use "Ninja" for building
ninja -t clean
```

---

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

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
