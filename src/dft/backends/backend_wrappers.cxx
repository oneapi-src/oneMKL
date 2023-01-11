/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
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

/*
This file lists functions matching those required by dft_function_table_t in 
src/dft/function_table.hpp.

To use this:

#define WRAPPER_VERSION <Wrapper version number>
#define BACKEND         <Backend name eg. mklgpu>

extern "C" dft_function_table_t mkl_dft_table = {
    WRAPPER_VERSION,
#include "dft/backends/backend_wrappers.cxx"
};

Changes to this file should be matched to changes in function_table.hpp. The required 
function template instantiations must be added to backend_backward_instantiations.cxx 
and backend_forward_instantiations.cxx.
*/

// clang-format off
oneapi::mkl::dft::BACKEND::create_commit,
oneapi::mkl::dft::BACKEND::create_commit,
oneapi::mkl::dft::BACKEND::create_commit,
oneapi::mkl::dft::BACKEND::create_commit,
#define ONEAPI_MKL_DFT_BACKEND_SIGNATURES(PRECISION, DOMAIN, T_REAL, T_FORWARD, T_BACKWARD)   \
    /* Buffer API */                                                                          \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>,                 \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>,      \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>,             \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_BACKWARD>,     \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL,  T_REAL>,            \
    /* USM API */                                                                             \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>,                 \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_FORWARD, T_BACKWARD>,      \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>,             \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_BACKWARD>,     \
    oneapi::mkl::dft::BACKEND::compute_forward<                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>,             \
    /* Buffer API */                                                                          \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>,                 \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>,      \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>,             \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_BACKWARD>,     \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>,             \
    /* USM API */                                                                             \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD>,                 \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL>,                     \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_FORWARD>,      \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>,             \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_BACKWARD, T_BACKWARD>,     \
    oneapi::mkl::dft::BACKEND::compute_backward<                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN>, T_REAL, T_REAL>,

ONEAPI_MKL_DFT_BACKEND_SIGNATURES(oneapi::mkl::dft::detail::precision::SINGLE,
                                  oneapi::mkl::dft::detail::domain::REAL, float, float,
                                  std::complex<float>)
ONEAPI_MKL_DFT_BACKEND_SIGNATURES(oneapi::mkl::dft::detail::precision::SINGLE,
                                  oneapi::mkl::dft::detail::domain::COMPLEX, float,
                                  std::complex<float>, std::complex<float>)
ONEAPI_MKL_DFT_BACKEND_SIGNATURES(oneapi::mkl::dft::detail::precision::DOUBLE,
                                  oneapi::mkl::dft::detail::domain::REAL, double,
                                  double, std::complex<double>)
ONEAPI_MKL_DFT_BACKEND_SIGNATURES(oneapi::mkl::dft::detail::precision::DOUBLE,
                                  oneapi::mkl::dft::detail::domain::COMPLEX, double,
                                  std::complex<double>, std::complex<double>)
// clang-format on

#undef ONEAPI_MKL_DFT_BACKEND_SIGNATURES
