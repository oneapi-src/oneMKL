/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef _DFT_FUNCTION_TABLE_HPP_
#define _DFT_FUNCTION_TABLE_HPP_

#include <complex>
#include <cstdint>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"

typedef struct {
    int version;
    oneapi::mkl::dft::detail::commit_impl<oneapi::mkl::dft::precision::SINGLE,
                                          oneapi::mkl::dft::domain::COMPLEX>* (
        *create_commit_sycl_fz)(
        const oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                           oneapi::mkl::dft::domain::COMPLEX>& desc,
        sycl::queue& sycl_queue);
    oneapi::mkl::dft::detail::commit_impl<oneapi::mkl::dft::precision::DOUBLE,
                                          oneapi::mkl::dft::domain::COMPLEX>* (
        *create_commit_sycl_dz)(
        const oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                           oneapi::mkl::dft::domain::COMPLEX>& desc,
        sycl::queue& sycl_queue);
    oneapi::mkl::dft::detail::commit_impl<oneapi::mkl::dft::precision::SINGLE,
                                          oneapi::mkl::dft::domain::REAL>* (*create_commit_sycl_fr)(
        const oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                           oneapi::mkl::dft::domain::REAL>& desc,
        sycl::queue& sycl_queue);
    oneapi::mkl::dft::detail::commit_impl<oneapi::mkl::dft::precision::DOUBLE,
                                          oneapi::mkl::dft::domain::REAL>* (*create_commit_sycl_dr)(
        const oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                           oneapi::mkl::dft::domain::REAL>& desc,
        sycl::queue& sycl_queue);

#define ONEAPI_MKL_DFT_BACKEND_SIGNATURES(EXT, PRECISION, DOMAIN, T_REAL, T_FORWARD, T_BACKWARD)   \
    void (*compute_forward_buffer_inplace_real_##EXT)(                                             \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & inout);                                                          \
    void (*compute_forward_buffer_inplace_complex_##EXT)(                                          \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_BACKWARD, 1> & inout);                                                      \
    void (*compute_forward_buffer_inplace_split_##EXT)(                                            \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & inout_re, sycl::buffer<T_REAL, 1> & inout_im);                   \
    void (*compute_forward_buffer_outofplace_##EXT)(                                               \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_FORWARD, 1> & in, sycl::buffer<T_BACKWARD, 1> & out);                       \
    void (*compute_forward_buffer_outofplace_real_##EXT)(                                          \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & in, sycl::buffer<T_REAL, 1> & out);                              \
    void (*compute_forward_buffer_outofplace_complex_##EXT)(                                       \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_BACKWARD, 1> & in, sycl::buffer<T_BACKWARD, 1> & out);                      \
    void (*compute_forward_buffer_outofplace_split_##EXT)(                                         \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & in_re, sycl::buffer<T_REAL, 1> & in_im,                          \
        sycl::buffer<T_REAL, 1> & out_re, sycl::buffer<T_REAL, 1> & out_im);                       \
    sycl::event (*compute_forward_usm_inplace_real_##EXT)(                                         \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout,            \
        const std::vector<sycl::event>& dependencies);                                             \
    sycl::event (*compute_forward_usm_inplace_complex_##EXT)(                                      \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,        \
        const std::vector<sycl::event>& dependencies);                                             \
    sycl::event (*compute_forward_usm_inplace_split_##EXT)(                                        \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re,         \
        T_REAL * inout_im, const std::vector<sycl::event>& dependencies);                          \
    sycl::event (*compute_forward_usm_outofplace_##EXT)(                                           \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_FORWARD * in,            \
        T_BACKWARD * out, const std::vector<sycl::event>& dependencies);                           \
    sycl::event (*compute_forward_usm_outofplace_real_##EXT)(                                      \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in, T_REAL * out, \
        const std::vector<sycl::event>& dependencies);                                             \
    sycl::event (*compute_forward_usm_outofplace_complex_##EXT)(                                   \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * in,           \
        T_BACKWARD * out, const std::vector<sycl::event>& dependencies);                           \
    sycl::event (*compute_forward_usm_outofplace_split_##EXT)(                                     \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re,            \
        T_REAL * in_im, T_REAL * out_re, T_REAL * out_im,                                          \
        const std::vector<sycl::event>& dependencies);                                             \
    void (*compute_backward_buffer_inplace_real_##EXT)(                                            \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & inout);                                                          \
    void (*compute_backward_buffer_inplace_complex_##EXT)(                                         \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_BACKWARD, 1> & inout);                                                      \
    void (*compute_backward_buffer_inplace_split_##EXT)(                                           \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & inout_re, sycl::buffer<T_REAL, 1> & inout_im);                   \
    void (*compute_backward_buffer_outofplace_##EXT)(                                              \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_BACKWARD, 1> & in, sycl::buffer<T_FORWARD, 1> & out);                       \
    void (*compute_backward_buffer_outofplace_real_##EXT)(                                         \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & in, sycl::buffer<T_REAL, 1> & out);                              \
    void (*compute_backward_buffer_outofplace_complex_##EXT)(                                      \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_BACKWARD, 1> & in, sycl::buffer<T_BACKWARD, 1> & out);                      \
    void (*compute_backward_buffer_outofplace_split_##EXT)(                                        \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc,                            \
        sycl::buffer<T_REAL, 1> & in_re, sycl::buffer<T_REAL, 1> & in_im,                          \
        sycl::buffer<T_REAL, 1> & out_re, sycl::buffer<T_REAL, 1> & out_im);                       \
    sycl::event (*compute_backward_usm_inplace_real_##EXT)(                                        \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout,            \
        const std::vector<sycl::event>& dependencies);                                             \
    sycl::event (*compute_backward_usm_inplace_complex_##EXT)(                                     \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * inout,        \
        const std::vector<sycl::event>& dependencies);                                             \
    sycl::event (*compute_backward_usm_inplace_split_##EXT)(                                       \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * inout_re,         \
        T_REAL * inout_im, const std::vector<sycl::event>& dependencies);                          \
    sycl::event (*compute_backward_usm_outofplace_##EXT)(                                          \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * in,           \
        T_FORWARD * out, const std::vector<sycl::event>& dependencies);                            \
    sycl::event (*compute_backward_usm_outofplace_real_##EXT)(                                     \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in, T_REAL * out, \
        const std::vector<sycl::event>& dependencies);                                             \
    sycl::event (*compute_backward_usm_outofplace_complex_##EXT)(                                  \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_BACKWARD * in,           \
        T_BACKWARD * out, const std::vector<sycl::event>& dependencies);                           \
    sycl::event (*compute_backward_usm_outofplace_split_##EXT)(                                    \
        oneapi::mkl::dft::detail::descriptor<PRECISION, DOMAIN> & desc, T_REAL * in_re,            \
        T_REAL * in_im, T_REAL * out_re, T_REAL * out_im,                                          \
        const std::vector<sycl::event>& dependencies);

    ONEAPI_MKL_DFT_BACKEND_SIGNATURES(f, oneapi::mkl::dft::detail::precision::SINGLE,
                                      oneapi::mkl::dft::detail::domain::REAL, float, float,
                                      std::complex<float>)
    ONEAPI_MKL_DFT_BACKEND_SIGNATURES(c, oneapi::mkl::dft::detail::precision::SINGLE,
                                      oneapi::mkl::dft::detail::domain::COMPLEX, float,
                                      std::complex<float>, std::complex<float>)
    ONEAPI_MKL_DFT_BACKEND_SIGNATURES(d, oneapi::mkl::dft::detail::precision::DOUBLE,
                                      oneapi::mkl::dft::detail::domain::REAL, double, double,
                                      std::complex<double>)
    ONEAPI_MKL_DFT_BACKEND_SIGNATURES(z, oneapi::mkl::dft::detail::precision::DOUBLE,
                                      oneapi::mkl::dft::detail::domain::COMPLEX, double,
                                      std::complex<double>, std::complex<double>)

#undef ONEAPI_MKL_DFT_BACKEND_SIGNATURES
} dft_function_table_t;

#endif //_DFT_FUNCTION_TABLE_HPP_
