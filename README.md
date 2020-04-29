# oneAPI Math Kernel Library (oneMKL) Interfaces

## Contents

- [Introduction](#introduction)
- [Support and Requirements](#support-and-requirements)
- [Build Setup](#build-setup)
- [Building with Conan](#building-with-conan)
- [Building with CMake](#building-with-cmake)
- [Project Cleanup](#project-cleanup)
<<<<<<< HEAD
- [FAQs](#faqs)
=======
>>>>>>> [README] Fixed links for legal notice
- [Legal Information](#legal-information)

---

## Introduction

oneMKL interfaces is an open-source implementation of oneMKL Data Parallel C++ (DPC++) interfaces according to [oneMKL specification](https://spec.oneapi.com/versions/latest/elements/oneMKL/source/index.html) that can work with multiple devices (backends) using device specific libraries underneath.

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

- **Run-time dispatching**: The application is linked with the onemkl library and the required backend is loaded at run-time based on device vendor (all libraries should be dynamic).

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

- **Compile-time dispatching**: The application uses a templated API where the template parameters specify the required backends and third-party libraries and the application is linked with required onemkl backend wrapper libraries (libraries can be static or dynamic).

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
 
---

## Support and Requirements

### Hardware Platform Support

#### Linux*
- CPU
    - Intel Atom(R) Processors
    - Intel(R) Core(TM) Processor Family
    - Intel(R) Xeon(R) Processor Family
- Accelerators
    - Intel(R) Processor Graphics GEN9
    - NVIDIA(R) TITAN RTX(TM) (Not tested with other NVIDIA GPU families and products.)

---
### Supported Operating Systems

#### Linux*

Operating System | CPU Host/Target | Integrated Graphics from Intel (Intel GPU) |  NVIDIA GPU
:--- | :--- | :--- | :---
Ubuntu                            | 18.04.3, 19.04 | 18.04.3, 19.10  | 18.04.3
SUSE Linux Enterprise Server*     | 15             | *Not supported* | *Not supported*
Red Hat Enterprise Linux* (RHEL*) | 8              | *Not supported* | *Not supported*
Linux* kernel                     | *N/A*          | 4.11 or higher | *N/A*

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
            <td colspan=4 align=center> GNU* GCC 5.1 or higher </td>
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
        <tr>
            <td rowspan=8> Linux* </td>
            <tr>
                <td rowspan=2> Intel CPU </td>
                <td> Intel(R) oneAPI DPC++ Compiler <br> or <br> Intel project for LLVM* technology </td>
                <td> No</td>
                <tr>
                    <td> Intel(R) oneAPI Math Kernel Library </td>
                    <td> Yes </td>
                </tr>
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
            <td rowspan=2> NVIDIA GPU </td>
            <td> Intel project for LLVM* technology </td>
            <td> No </td>
            <tr>
                <td> NVIDIA CUDA SDK </td>
                <td> No </td>
            </tr>
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

Product | Supported Version | Conan Package Source | Conan Package Install Location | License
:--- | :--- | :--- | :--- | :---
Python | 3.6 or higher | *Pre-installed* | *Pre-installed* | [PSF](https://docs.python.org/3.6/license.html)
[Conan C++ Package Manager](https://conan.io/downloads.html) | 1.23 or higher | *Installed by user* | *Installed by user* | [MIT](https://github.com/conan-io/conan/blob/develop/LICENSE.md)
[CMake](https://cmake.org/download/) | 3.13 or higher | conan-center | ~/.conan/data or $CONAN_USER_HOME/.conan/data | [The OSI-approved BSD 3-clause License](https://gitlab.kitware.com/cmake/cmake/raw/master/Copyright.txt)
[Ninja](https://ninja-build.org/) | 1.9.0 | conan-center | ~/.conan/data or $CONAN_USER_HOME/.conan/data | [Apache License v2.0](https://github.com/ninja-build/ninja/blob/master/COPYING)
[GNU* FORTRAN Compiler](https://gcc.gnu.org/wiki/GFortran) | 7.4.0 or higher | apt | /usr/bin | [GNU General Public License, version 3](https://gcc.gnu.org/onlinedocs/gcc-7.5.0/gfortran/Copying.html)
[Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) | 2021.1-beta05 | *Installed by user* | *Installed by user* | [End User License Agreement for the Intel(R) Software Development Products](https://software.intel.com/en-us/license/eula-for-intel-software-development-products)
[Intel project for LLVM* technology binary for Intel CPU](https://github.com/intel/llvm/releases) | Daily builds (experimental) tested with [20200331](https://github.com/intel/llvm/releases/download/20200331/dpcpp-compiler.tar.gz) | *Installed by user* | *Installed by user* | [Apache License v2](https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT)
[Intel(R) oneAPI Math Kernel Library](https://software.intel.com/en-us/oneapi/onemkl) | 2021.1-beta05 | apt | /opt/intel/inteloneapi/mkl | [Intel Simplified Software License](https://software.intel.com/en-us/license/intel-simplified-software-license)
[NVIDIA CUDA SDK](https://developer.nvidia.com/cublas) | 10.2 | *Installed by user* | *Installed by user* |[End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html)
[NETLIB LAPACK](https://www.netlib.org/) | 3.7.1 | conan-community | ~/.conan/data or $CONAN_USER_HOME/.conan/data | [BSD like license](http://www.netlib.org/lapack/LICENSE.txt)
[Sphinx](https://www.sphinx-doc.org/en/master/) | 2.4.4 | pip | ~/.local/bin (or similar user local directory) | [BSD License](https://github.com/sphinx-doc/sphinx/blob/3.x/LICENSE)

*conan-center: https://api.bintray.com/conan/conan/conan-center*

*conan-community: https://api.bintray.com/conan/conan-community/conan*

---

## Build Setup

1. Install Intel(R) oneAPI DPC++ Compiler (select variant as per requirement).

2. Clone this project to `<path to onemkl>`, where `<path to onemkl>` is the root directory of this repository.

3. You can [Build with Conan](#building-with-conan) to automate the process of getting dependencies or you can download and install the required dependencies manually and [Build with CMake](#building-with-cmake) directly .

*Note: Conan package manager automates the process of getting required packages, so that you do not have to go to different web location and follow different instructions to install them.*

---

## Building with Conan

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

- Intel(R) oneAPI DPC++ Compiler for Intel CPU and Intel GPU backend: `inteldpcpp_lnx`

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
- `enable_mklcpu_backend=[True | False]`. Setting it to `True` enables the building of oneMKL intelmkl cpu backend. The default value is `True`.
- `enable_mklgpu_backend=[True | False]`. Setting it to `True` enables the building of oneMKL intelmkl gpu backend. The default value is `True`.
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

### Build Options
All options specified in the Conan section are available to CMake. You can specify these options using `-D<cmake_option>=<value>`.

The following table provides a detailed mapping of options between Conan and CMake.

Conan Option | CMake Option | Supported Values | Default Value
 :---------- | :----------- | :--------------- | :---
build_shared_libs        | BUILD_SHARED_LIBS        | True, False         | True
enable_mklcpu_backend    | ENABLE_MKLCPU_BACKEND    | True, False         | True
enable_mklgpu_backend    | ENABLE_MKLGPU_BACKEND    | True, False         | True
*Not Supported*          | ENABLE_CUBLAS_BACKEND    | True, False         | False
enable_mklcpu_thread_tbb | ENABLE_MKLCPU_THREAD_TBB | True, False         | True
build_functional_tests   | BUILD_FUNCTIONAL_TESTS   | True, False         | True
build_doc                | BUILD_DOC                | True, False         | False

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

## FAQs

### Conan

1. I am behind a proxy. How can Conan download dependencies from external network?
   - `~/.conan/conan.conf` has a `[proxies]` section where you can add the list of proxies. For details refer to [Conan proxy settings](https://docs.conan.io/en/latest/reference/config_files/conan.conf.html#proxies).

---

#### [Legal information](legal_information.md)
