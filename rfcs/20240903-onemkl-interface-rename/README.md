# OneMKL Interface renaming

### Revision


|Date       |Revision | Comments                                                                 |
|-----------|---------|--------------------------------------------------------------------------|
|  20240903 |  1.0    | Initial version                                                          |

## Motivation

As oneMKL interface is moving to the UXL foundation we should make sure that the
name does not collide with the existing [Intel oneMKL
product](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).

We have been discussing 2 solutions to avoid this issue:
1. Renaming the [oneMKL](https://github.com/oneapi-src/oneMKL) GitHub project to
   a new name to be discussed and replacing or clarifying the occurrences of
   "oneMKL" in that repository.
2. Using this opportunity to also split this GitHub project per domain. In this
   case the names that are advertised would be based on each domain of oneMKL
   like `one<Domain>`.

We think that the first solution should be enough to answer the main concern. It
is not clear whether the second solution could better answer the need of users
so we are not planning to split oneMKL per domain unless there is clear and
strong feedback from users.

**We aim to agree on a solution by October 4, 2024 and have the proposal
implemented by the end of November 2024.**

There has been a number issues of where we have had to clarify the differences
between oneMKL Interface and the Intel oneMKL product
[#501](https://github.com/oneapi-src/oneMKL/issues/501#issuecomment-2134681621);
[#377](https://github.com/oneapi-src/oneMKL/issues/377);
[#253](https://github.com/oneapi-src/oneMKL/issues/253);
[#222](https://github.com/oneapi-src/oneMKL/issues/222);
[#211](https://github.com/oneapi-src/oneMKL/issues/211);
[#207](https://github.com/oneapi-src/oneMKL/issues/207);
[#206](https://github.com/oneapi-src/oneMKL/issues/206). Many of these issues
are from 2022. I believe this is still an issue today since this has been an
issue on multiple occasion internally when Codeplay started to contribute to
oneMKL Interface. The oneMKL Interface
[README](https://github.com/oneapi-src/oneMKL?tab=readme-ov-file#onemkl) already
explains the differences between oneMKL Interface and Intel oneMKL product but
this is not enough for people who are not working directly on the projects.

## Outline

1. [Introduction](#introduction)
2. [Proposal](#proposal)
3. [User impact](#user-impact)
4. [Open questions](#open-questions)

## Introduction

oneMKL Interface has historically been implemented and supported by Intel. As
the project is moving to the UXL foundation we want to avoid using the name of
an Intel product. This RFC describes the solution of renaming oneMKL Interface
to a new name.

## Proposal

The main purpose of this RFC is to agree on a new name for the oneMKL specification and oneMKL Interface.
Some of the suggested names for the implementations are:
* **oneMath**
  * The specification would be renamed oneMath Specification
* **oneSLA** (SYCL Linear Algebra)
  * The specification would be renamed oneSLA Specification

Other suggestions are welcomed. The name **oneMath** will be chosen if there are
no objections by October 4, 2024.

The suggested solution is to proceed in the following steps:
1. The UXL foundation agrees on the new name.
2. Codeplay submits a oneAPI-spec PR to rename the occurrences of "oneMKL" to
   the new name.
3. Codeplay submits a oneMKL Interface PR to:
   * Update the root README to use the new name, with a mention that the project
     was formerly called oneMKL Interface.
   * Update the references to "oneMKL" and `onemkl_` in the documentation as
     seen in the first few lines of
     [docs/onemkl-datatypes.rst](https://github.com/oneapi-src/oneMKL/blob/develop/docs/onemkl-datatypes.rst?plain=1#L1)
     for instance.
   * Update occurrences of "onemkl" in internal functions such as
     [onemkl_cublas_host_task](https://github.com/oneapi-src/oneMKL/blob/1ce98a699f93bd3a78350269b2e34d822fe43b91/src/blas/backends/cublas/cublas_task.hpp#L77).
   * Update macros such as include guards and other internal macros like
     `ONEMKL_EXPORT` to use the new name.
   * Rename CMake targets `onemkl` and `onemkl_<domain>_<backend>` to use the
     new name. The existing targets name can be added with a deprecation
     messages for anyone using them. See the section on [CMake target
     deprecation](#cmake-deprecated-target) for more details.
4. Once the PRs are approved, Codeplay transfers the
   [oneMKL](https://github.com/oneapi-src/oneMKL) GitHub project to the
   [uxlfoundation](https://github.com/uxlfoundation) organization under the new
   name. We use the
   [transferring](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository)
   feature from GitHub so the links from the previous repository are redirected
   to the new one.
5. The PRs from the step 2 and 3 are merged.

We are not planning to rename the occurrences of "mkl" such as the `oneapi::mkl`
namespace, the `include/oneapi/mkl` folder or the `include/oneapi/mkl.hpp` file.
Whether this is needed is an open question.

### CMake target deprecation

CMake allows to set a
[`DEPRECATION`](https://cmake.org/cmake/help/latest/prop_tgt/DEPRECATION.html)
property on a target which will print a custom message whenever the target is
used. The property cannot be set on an [alias
target](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#alias-targets)
as they are read-only. The property can be set on an imported target instead
like the example below:

```cmake
add_library(onemath lib.cpp) # New main target, for the example

add_library(onemkl INTERFACE IMPORTED) # onemkl works like an alias of onemath which can have different properties
target_link_libraries(onemkl INTERFACE onemath)
set_target_properties(onemkl PROPERTIES DEPRECATION "onemkl target is deprecated, please use onemath instead")

add_executable(main main.cpp)
target_link_libraries(main PUBLIC onemkl) # Prints a warning at CMake configuration time
```

The same solution can be used for the `onemkl_<domain>_<backend>` targets. This does
not add any extra targets to the generated `Makefile` or `build.ninja` files so
the library will not be built twice.

### Other Considered Approaches

Another considered approach is to split the existing oneMKL Interface per domain
like so: oneBLAS, oneLAPACK, oneDFT, oneRNG, oneSPARSE. This shifts the need of
renaming "oneMKL Interface" as the main visible names will be based on the
domain. It is not clear whether this better answers the users needs.

With this approach the suggested solution is to have a common repository for
common types (`transpose`, `uplo`, `diag`, `side`, `offset`, `index_base`,
`layout`), exceptions and some CMake logic. Each domain would have its own
repository automatically pulling the common headers. Another repository could be
created to automatically pull multiple domains which would mimic the behavior of
the current oneMKL Interface.

We are not planning to proceed with this approach unless users express a strong
preference.

## User impact

The suggested solution does not break any existing code.
* The repository is transferred using the [GitHub
  transfer](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository)
  feature so users accessing or pulling from
  https://github.com/oneapi-src/oneMKL will be redirected to the new link.
* The changes in oneMKL do not affect the public API. The macros renamed or the
  header files in the `detail` folders renamed should not be used outside of the
  oneMKL Interface project.
* The CMake changes will still provide the same targets but will print a warning
  message if users use targets with the `onemkl` name.

## Open questions

* Other suggestions for new names are welcomed.
* Is it needed to rename the occurrences of "mkl"?
   * This will have a bigger impact and require more time to complete.
   * It should be possible to rename these occurrences without any breaking
     change. This would need to be further investigated.
* Should the specification and the existing implementation have different names?
  * Currently both the specification and implementation are named based on
    "oneMKL" which suggests that the oneMKL Interface is the main or only
    implementation of the specification. If the goal of the specification is to
    encourage for multiple implementations to co-exist then it should be named
    differently than the implementation.
  * Given the nature of the project that allows for multiple backends I don't
    see any value in encouraging multiple implementations as of today.
  * Using multiple names may create more confusion.
