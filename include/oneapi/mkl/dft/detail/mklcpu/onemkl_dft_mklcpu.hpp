/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

#define ONEAPI_MKL_DFT_BACKEND_SIGNATURES(EXT, PRECISION, DOMAIN, T_REAL, T_FORWARD, T_BACKWARD)        \
                                                                                                        \
    void commit_##EXT(descriptor<PRECISION, DOMAIN> &desc, sycl::queue &queue);                         \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform*/                                                                              \
    void compute_forward_buffer_inplace_##EXT(descriptor<PRECISION, DOMAIN> &desc,                      \
                                              sycl::buffer<T_BACKWARD, 1> &inout);                      \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    void compute_forward_buffer_inplace_split_##EXT(descriptor<PRECISION, DOMAIN> &desc,                \
                                                    sycl::buffer<T_REAL, 1> &inout_re,                  \
                                                    sycl::buffer<T_REAL, 1> &inout_im);                 \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    void compute_forward_buffer_outofplace_##EXT(descriptor<PRECISION, DOMAIN> &desc,                   \
                                                 sycl::buffer<T_FORWARD, 1> &in,                        \
                                                 sycl::buffer<T_BACKWARD, 1> &out);                     \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    void compute_forward_buffer_outofplace_split_##EXT(                                                 \
        descriptor<PRECISION, DOMAIN> &desc, sycl::buffer<T_REAL, 1> &in_re,                            \
        sycl::buffer<T_REAL, 1> &in_im, sycl::buffer<T_REAL, 1> &out_re,                                \
        sycl::buffer<T_REAL, 1> &out_im);                                                               \
                                                                                                        \
    /*USM version*/                                                                                     \
                                                                                                        \
    /*In-place transform*/                                                                              \
    sycl::event compute_forward_usm_inplace_##EXT(                                                      \
        descriptor<PRECISION, DOMAIN> &desc, T_BACKWARD *inout,                                         \
        const std::vector<sycl::event> &dependencies = {});                                             \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    sycl::event compute_forward_usm_inplace_split_##EXT(                                                \
        descriptor<PRECISION, DOMAIN> &desc, T_REAL *inout_re, T_REAL *inout_im,                        \
        const std::vector<sycl::event> &dependencies = {});                                             \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    sycl::event compute_forward_usm_outofplace_##EXT(                                                   \
        descriptor<PRECISION, DOMAIN> &desc, T_FORWARD *in, T_BACKWARD *out,                            \
        const std::vector<sycl::event> &dependencies = {});                                             \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    sycl::event compute_forward_usm_outofplace_split_##EXT(                                             \
        descriptor<PRECISION, DOMAIN> &desc, T_REAL *in_re, T_REAL *in_im, T_REAL *out_re,              \
        T_REAL *out_im, const std::vector<sycl::event> &dependencies = {});                             \
                                                                                                        \
    /*Buffer version*/                                                                                  \
                                                                                                        \
    /*In-place transform*/                                                                              \
    void compute_backward_buffer_inplace_##EXT(descriptor<PRECISION, DOMAIN> &desc,                     \
                                               sycl::buffer<T_BACKWARD, 1> &inout);                     \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    void compute_backward_buffer_inplace_split_##EXT(descriptor<PRECISION, DOMAIN> &desc,               \
                                                     sycl::buffer<T_REAL, 1> &inout_re,                 \
                                                     sycl::buffer<T_REAL, 1> &inout_im);                \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    void compute_backward_buffer_outofplace_##EXT(descriptor<PRECISION, DOMAIN> &desc,                  \
                                                  sycl::buffer<T_BACKWARD, 1> &in,                      \
                                                  sycl::buffer<T_FORWARD, 1> &out);                     \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    void compute_backward_buffer_outofplace_split_##EXT(                                                \
        descriptor<PRECISION, DOMAIN> &desc, sycl::buffer<T_REAL, 1> &in_re,                            \
        sycl::buffer<T_REAL, 1> &in_im, sycl::buffer<T_REAL, 1> &out_re,                                \
        sycl::buffer<T_REAL, 1> &out_im);                                                               \
                                                                                                        \
    /*USM version*/                                                                                     \
                                                                                                        \
    /*In-place transform*/                                                                              \
    sycl::event compute_backward_usm_inplace_##EXT(                                                     \
        descriptor<PRECISION, DOMAIN> &desc, T_BACKWARD *inout,                                         \
        const std::vector<sycl::event> &dependencies = {});                                             \
                                                                                                        \
    /*In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/     \
    sycl::event compute_backward_usm_inplace_split_##EXT(                                               \
        descriptor<PRECISION, DOMAIN> &desc, T_REAL *inout_re, T_REAL *inout_im,                        \
        const std::vector<sycl::event> &dependencies = {});                                             \
                                                                                                        \
    /*Out-of-place transform*/                                                                          \
    sycl::event compute_backward_usm_outofplace_##EXT(                                                  \
        descriptor<PRECISION, DOMAIN> &desc, T_BACKWARD *in, T_FORWARD *out,                            \
        const std::vector<sycl::event> &dependencies = {});                                             \
                                                                                                        \
    /*Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format*/ \
    sycl::event compute_backward_usm_outofplace_split_##EXT(                                            \
        descriptor<PRECISION, DOMAIN> &desc, T_REAL *in_re, T_REAL *in_im, T_REAL *out_re,              \
        T_REAL *out_im, const std::vector<sycl::event> &dependencies = {});

ONEAPI_MKL_DFT_BACKEND_SIGNATURES(f, precision::SINGLE, domain::REAL, float, float,
                                  std::complex<float>)
ONEAPI_MKL_DFT_BACKEND_SIGNATURES(c, precision::SINGLE, domain::COMPLEX, float, std::complex<float>,
                                  std::complex<float>)
ONEAPI_MKL_DFT_BACKEND_SIGNATURES(d, precision::DOUBLE, domain::REAL, double, double,
                                  std::complex<double>)
ONEAPI_MKL_DFT_BACKEND_SIGNATURES(z, precision::DOUBLE, domain::COMPLEX, double,
                                  std::complex<double>, std::complex<double>)

#undef ONEAPI_MKL_DFT_BACKEND_SIGNATURES

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi
