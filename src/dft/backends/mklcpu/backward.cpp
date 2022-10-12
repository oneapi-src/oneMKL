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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/types.hpp"

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

void compute_backward_buffer_inplace_f(descriptor<precision::SINGLE, domain::REAL> &desc,
                                       sycl::buffer<std::complex<float>, 1> &inout) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_inplace_c(descriptor<precision::SINGLE, domain::COMPLEX> &desc,
                                       sycl::buffer<std::complex<float>, 1> &inout) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_inplace_d(descriptor<precision::DOUBLE, domain::REAL> &desc,
                                       sycl::buffer<std::complex<double>, 1> &inout) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_inplace_z(descriptor<precision::DOUBLE, domain::COMPLEX> &desc,
                                       sycl::buffer<std::complex<double>, 1> &inout) {
    throw std::runtime_error("Not implemented for mklcpu");
}

void compute_backward_buffer_inplace_split_f(descriptor<precision::SINGLE, domain::REAL> &desc,
                                             sycl::buffer<float, 1> &inout_re,
                                             sycl::buffer<float, 1> &inout_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_inplace_split_c(descriptor<precision::SINGLE, domain::COMPLEX> &desc,
                                             sycl::buffer<float, 1> &inout_re,
                                             sycl::buffer<float, 1> &inout_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_inplace_split_d(descriptor<precision::DOUBLE, domain::REAL> &desc,
                                             sycl::buffer<double, 1> &inout_re,
                                             sycl::buffer<double, 1> &inout_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_inplace_split_z(descriptor<precision::DOUBLE, domain::COMPLEX> &desc,
                                             sycl::buffer<double, 1> &inout_re,
                                             sycl::buffer<double, 1> &inout_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}

void compute_backward_buffer_outofplace_f(descriptor<precision::SINGLE, domain::REAL> &desc,
                                          sycl::buffer<std::complex<float>, 1> &in,
                                          sycl::buffer<float, 1> &out) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_outofplace_c(descriptor<precision::SINGLE, domain::COMPLEX> &desc,
                                          sycl::buffer<std::complex<float>, 1> &in,
                                          sycl::buffer<std::complex<float>, 1> &out) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_outofplace_d(descriptor<precision::DOUBLE, domain::REAL> &desc,
                                          sycl::buffer<std::complex<double>, 1> &in,
                                          sycl::buffer<double, 1> &out) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_outofplace_z(descriptor<precision::DOUBLE, domain::COMPLEX> &desc,
                                          sycl::buffer<std::complex<double>, 1> &in,
                                          sycl::buffer<std::complex<double>, 1> &out) {
    throw std::runtime_error("Not implemented for mklcpu");
}

void compute_backward_buffer_outofplace_split_f(descriptor<precision::SINGLE, domain::REAL> &desc,
                                                sycl::buffer<float, 1> &in_re,
                                                sycl::buffer<float, 1> &in_im,
                                                sycl::buffer<float, 1> &out_re,
                                                sycl::buffer<float, 1> &out_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_outofplace_split_c(
    descriptor<precision::SINGLE, domain::COMPLEX> &desc, sycl::buffer<float, 1> &in_re,
    sycl::buffer<float, 1> &in_im, sycl::buffer<float, 1> &out_re, sycl::buffer<float, 1> &out_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_outofplace_split_d(descriptor<precision::DOUBLE, domain::REAL> &desc,
                                                sycl::buffer<double, 1> &in_re,
                                                sycl::buffer<double, 1> &in_im,
                                                sycl::buffer<double, 1> &out_re,
                                                sycl::buffer<double, 1> &out_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}
void compute_backward_buffer_outofplace_split_z(
    descriptor<precision::DOUBLE, domain::COMPLEX> &desc, sycl::buffer<double, 1> &in_re,
    sycl::buffer<double, 1> &in_im, sycl::buffer<double, 1> &out_re,
    sycl::buffer<double, 1> &out_im) {
    throw std::runtime_error("Not implemented for mklcpu");
}

sycl::event compute_backward_usm_inplace_f(descriptor<precision::SINGLE, domain::REAL> &desc,
                                           std::complex<float> *inout,
                                           const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_inplace_c(descriptor<precision::SINGLE, domain::COMPLEX> &desc,
                                           std::complex<float> *inout,
                                           const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_inplace_d(descriptor<precision::DOUBLE, domain::REAL> &desc,
                                           std::complex<double> *inout,
                                           const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_inplace_z(descriptor<precision::DOUBLE, domain::COMPLEX> &desc,
                                           std::complex<double> *inout,
                                           const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}

sycl::event compute_backward_usm_inplace_split_f(descriptor<precision::SINGLE, domain::REAL> &desc,
                                                 float *inout_re, float *inout_im,
                                                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_inplace_split_c(
    descriptor<precision::SINGLE, domain::COMPLEX> &desc, float *inout_re, float *inout_im,
    const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_inplace_split_d(descriptor<precision::DOUBLE, domain::REAL> &desc,
                                                 double *inout_re, double *inout_im,
                                                 const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_inplace_split_z(
    descriptor<precision::DOUBLE, domain::COMPLEX> &desc, double *inout_re, double *inout_im,
    const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}

sycl::event compute_backward_usm_outofplace_f(descriptor<precision::SINGLE, domain::REAL> &desc,
                                              std::complex<float> *in, float *out,
                                              const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_outofplace_c(descriptor<precision::SINGLE, domain::COMPLEX> &desc,
                                              std::complex<float> *in, std::complex<float> *out,
                                              const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_outofplace_d(descriptor<precision::DOUBLE, domain::REAL> &desc,
                                              std::complex<double> *in, double *out,
                                              const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_outofplace_z(descriptor<precision::DOUBLE, domain::COMPLEX> &desc,
                                              std::complex<double> *in, std::complex<double> *out,
                                              const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}

sycl::event compute_backward_usm_outofplace_split_f(
    descriptor<precision::SINGLE, domain::REAL> &desc, float *in_re, float *in_im, float *out_re,
    float *out_im, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_outofplace_split_c(
    descriptor<precision::SINGLE, domain::COMPLEX> &desc, float *in_re, float *in_im, float *out_re,
    float *out_im, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_outofplace_split_d(
    descriptor<precision::DOUBLE, domain::REAL> &desc, double *in_re, double *in_im, double *out_re,
    double *out_im, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}
sycl::event compute_backward_usm_outofplace_split_z(
    descriptor<precision::DOUBLE, domain::COMPLEX> &desc, double *in_re, double *in_im,
    double *out_re, double *out_im, const std::vector<sycl::event> &dependencies) {
    throw std::runtime_error("Not implemented for mklcpu");
}

} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi
