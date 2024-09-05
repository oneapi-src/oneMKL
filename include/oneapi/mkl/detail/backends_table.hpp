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

enum class device : uint16_t { x86cpu, intelgpu, nvidiagpu, amdgpu, generic_device };
enum class domain : uint16_t { blas, dft, lapack, rng, sparse_blas };

static std::map<domain, std::map<device, std::vector<const char*>>> libraries = {
    { domain::blas,
      { { device::x86cpu,
          {
#ifdef ENABLE_MKLCPU_BACKEND
              LIB_NAME("blas_mklcpu"),
#endif
#ifdef ENABLE_NETLIB_BACKEND
              LIB_NAME("blas_netlib"),
#endif
#ifdef ENABLE_PORTBLAS_BACKEND_INTEL_CPU
              LIB_NAME("blas_portblas"),
#endif
          } },
        { device::intelgpu,
          {
#ifdef ENABLE_MKLGPU_BACKEND
              LIB_NAME("blas_mklgpu"),
#endif
#ifdef ENABLE_PORTBLAS_BACKEND_INTEL_GPU
              LIB_NAME("blas_portblas"),
#endif
          } },
        { device::amdgpu,
          {
#ifdef ENABLE_ROCBLAS_BACKEND
              LIB_NAME("blas_rocblas"),
#endif
#ifdef ENABLE_PORTBLAS_BACKEND_AMD_GPU
              LIB_NAME("blas_portblas"),
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ENABLE_CUBLAS_BACKEND
              LIB_NAME("blas_cublas"),
#endif
#ifdef ENABLE_PORTBLAS_BACKEND_NVIDIA_GPU
              LIB_NAME("blas_portblas"),
#endif
          } },
        { device::generic_device,
          {
#ifdef ENABLE_PORTBLAS_BACKEND
              LIB_NAME("blas_portblas"),
#endif
          } } } },

    { domain::dft,
      { { device::x86cpu,
          {
#ifdef ENABLE_MKLCPU_BACKEND
              LIB_NAME("dft_mklcpu")
#endif
#ifdef ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ENABLE_MKLGPU_BACKEND
              LIB_NAME("dft_mklgpu")
#endif
#ifdef ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ENABLE_ROCFFT_BACKEND
              LIB_NAME("dft_rocfft")
#endif
#ifdef ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ENABLE_CUFFT_BACKEND
              LIB_NAME("dft_cufft")
#endif
#ifdef ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
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
          } } } },

    { domain::sparse_blas,
      { { device::x86cpu,
          {
#ifdef ENABLE_MKLCPU_BACKEND
              LIB_NAME("sparse_blas_mklcpu")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ENABLE_MKLGPU_BACKEND
              LIB_NAME("sparse_blas_mklgpu")
#endif
          } } } },
};

static std::map<domain, const char*> table_names = { { domain::blas, "mkl_blas_table" },
                                                     { domain::lapack, "mkl_lapack_table" },
                                                     { domain::dft, "mkl_dft_table" },
                                                     { domain::rng, "mkl_rng_table" },
                                                     { domain::sparse_blas,
                                                       "mkl_sparse_blas_table" } };

} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_BACKENDS_TABLE_HPP_
