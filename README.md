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
            <td rowspan=5 align="center">oneMKL interface</td>
            <td rowspan=5 align="center">oneMKL selector</td>
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
            <td align="center"><a href="https://developer.nvidia.com/curand"> NVIDIA cuRAND</a> for NVIDIA GPU </td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://ww.netlib.org"> NETLIB LAPACK</a> for x86 CPU </td>
            <td align="center">x86 CPU</td>
        </tr>
    </tbody>
</table>

## Table of Contents

- [Documentation](#documentation)
- [FAQs](#faqs)
- [Legal Information](#legal-information)

---

## Documentation
- [Contents](https://oneapi-src.github.io/oneMKL/)
- [About](https://oneapi-src.github.io/oneMKL/introduction.html)
- [Get Started](https://oneapi-src.github.io/oneMKL/selecting_a_compiler.html)
- [Developer Reference](https://oneapi-src.github.io/oneMKL/onemkl-datatypes.html)

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
