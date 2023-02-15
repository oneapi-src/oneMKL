/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _ONEMKL_BACKENDS_TABLE_HPP_
#define _ONEMKL_BACKENDS_TABLE_HPP_

#include <string>
#include <vector>
#include <map>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/detail/config.hpp"

#ifdef __linux__
#define LIB_NAME(a) "libonemkl_" a ".so"
#elif defined(_WIN64)
#define LIB_NAME(a) "onemkl_" a ".dll"
#endif

namespace oneapi {
namespace mkl {

enum class device : uint16_t { x86cpu, intelgpu, nvidiagpu, amdgpu };
enum class domain : uint16_t { blas, lapack, rng };

static std::map<domain, std::map<device, std::vector<const char*>>> libraries = {
    { domain::blas,
      { { device::x86cpu,
          {
#ifdef ENABLE_MKLCPU_BACKEND
              LIB_NAME("blas_mklcpu"),
#endif
#ifdef ENABLE_NETLIB_BACKEND
              LIB_NAME("blas_netlib")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ENABLE_MKLGPU_BACKEND
              LIB_NAME("blas_mklgpu")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ENABLE_ROCBLAS_BACKEND
              LIB_NAME("blas_rocblas")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ENABLE_CUBLAS_BACKEND
              LIB_NAME("blas_cublas")
#endif
          } } } },

    { domain::lapack,
      { { device::x86cpu,
          {
#ifdef ENABLE_MKLCPU_BACKEND
              LIB_NAME("lapack_mklcpu")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ENABLE_MKLGPU_BACKEND
              LIB_NAME("lapack_mklgpu")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ENABLE_ROCSOLVER_BACKEND
              LIB_NAME("lapack_rocsolver")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ENABLE_CUSOLVER_BACKEND
              LIB_NAME("lapack_cusolver")
#endif
          } } } },

    { domain::rng,
      { { device::x86cpu,
          {
#ifdef ENABLE_MKLCPU_BACKEND
              LIB_NAME("rng_mklcpu")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ENABLE_MKLGPU_BACKEND
              LIB_NAME("rng_mklgpu")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ENABLE_ROCRAND_BACKEND
              LIB_NAME("rng_rocrand")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ENABLE_CURAND_BACKEND
              LIB_NAME("rng_curand")
#endif
          } } } }
};

static std::map<domain, const char*> table_names = { { domain::blas, "mkl_blas_table" },
                                                     { domain::lapack, "mkl_lapack_table" },
                                                     { domain::rng, "mkl_rng_table" } };

} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_BACKENDS_TABLE_HPP_
