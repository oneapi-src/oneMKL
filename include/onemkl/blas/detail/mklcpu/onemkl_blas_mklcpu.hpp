/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef _ONEMKL_BLAS_MKLCPU_HPP_
#define _ONEMKL_BLAS_MKLCPU_HPP_

#include <CL/sycl.hpp>

#include <complex>
#include <cstdint>

#include <onemkl/types.hpp>
#include "onemkl/detail/export.hpp"

namespace onemkl {

using onemkl::transpose;
using onemkl::uplo;
using onemkl::side;
using onemkl::diag;
using onemkl::offset;

namespace mklcpu {

// Level 1

ONEMKL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);

ONEMKL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result);

ONEMKL_EXPORT void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &result);

ONEMKL_EXPORT void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);

ONEMKL_EXPORT void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);

ONEMKL_EXPORT void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);

ONEMKL_EXPORT void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);

ONEMKL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);

ONEMKL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);

ONEMKL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result);

ONEMKL_EXPORT void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result);

ONEMKL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                       float s);

ONEMKL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                       double s);

ONEMKL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c,
                       float s);

ONEMKL_EXPORT void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       double c, double s);

ONEMKL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<float, 1> &s);

ONEMKL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<double, 1> &s);

ONEMKL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<std::complex<float>, 1> &s);

ONEMKL_EXPORT void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &b,
                        cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<std::complex<double>, 1> &s);

ONEMKL_EXPORT void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &param);

ONEMKL_EXPORT void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &param);

ONEMKL_EXPORT void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                         cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
                         cl::sycl::buffer<float, 1> &param);

ONEMKL_EXPORT void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                         cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                         double y1, cl::sycl::buffer<double, 1> &param);

ONEMKL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &result);

ONEMKL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

// Level 2

ONEMKL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);

ONEMKL_EXPORT void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);

ONEMKL_EXPORT void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);

ONEMKL_EXPORT void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);

ONEMKL_EXPORT void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);

ONEMKL_EXPORT void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);

ONEMKL_EXPORT void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a);

ONEMKL_EXPORT void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a);

ONEMKL_EXPORT void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a);

ONEMKL_EXPORT void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a);

ONEMKL_EXPORT void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                        std::int64_t incy);

ONEMKL_EXPORT void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a);

ONEMKL_EXPORT void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a);

ONEMKL_EXPORT void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a);

ONEMKL_EXPORT void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a);

ONEMKL_EXPORT void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);

ONEMKL_EXPORT void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda);

ONEMKL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);

ONEMKL_EXPORT void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);

ONEMKL_EXPORT void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);

ONEMKL_EXPORT void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx);

ONEMKL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);

ONEMKL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);

ONEMKL_EXPORT void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx);

// Level 3

ONEMKL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                        cl::sycl::buffer<half, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                         double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                         std::int64_t ldc);

ONEMKL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                        double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);

ONEMKL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                         float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                         double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                         std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                         std::int64_t ldc);

ONEMKL_EXPORT void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                         std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                         std::int64_t ldc);

ONEMKL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

ONEMKL_EXPORT void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

// Batch API

ONEMKL_EXPORT void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                              cl::sycl::buffer<transpose, 1> &transb,
                              cl::sycl::buffer<std::int64_t, 1> &m,
                              cl::sycl::buffer<std::int64_t, 1> &n,
                              cl::sycl::buffer<std::int64_t, 1> &k,
                              cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                              cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                              cl::sycl::buffer<std::int64_t, 1> &ldb,
                              cl::sycl::buffer<float, 1> &beta, cl::sycl::buffer<float, 1> &c,
                              cl::sycl::buffer<std::int64_t, 1> &ldc, std::int64_t group_count,
                              cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
    cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<float>, 1> &alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<float>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<float>, 1> &beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, float beta,
                              cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

ONEMKL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, double beta,
                              cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

ONEMKL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                              cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

ONEMKL_EXPORT void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size);

ONEMKL_EXPORT void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                              cl::sycl::buffer<uplo, 1> &upper_lower,
                              cl::sycl::buffer<transpose, 1> &trans,
                              cl::sycl::buffer<diag, 1> &unit_diag,
                              cl::sycl::buffer<std::int64_t, 1> &m,
                              cl::sycl::buffer<std::int64_t, 1> &n,
                              cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                              cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                              cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
                              cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
    cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<float>, 1> &alpha,
    cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<double>, 1> &alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);

ONEMKL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

ONEMKL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

ONEMKL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

ONEMKL_EXPORT void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

// BLAS-like extensions

ONEMKL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k, float alpha,
                         cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k, double alpha,
                         cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                         cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                         std::int64_t ldb, std::complex<float> beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, std::complex<double> beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                            cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta,
                            cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                            float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                            int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                            uint8_t bo, float beta, cl::sycl::buffer<int32_t, 1> &c,
                            std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);

ONEMKL_EXPORT void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                            cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                            cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                            cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                            cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k,
                            std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                            std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                            std::int64_t ldb, std::complex<float> beta,
                            cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

ONEMKL_EXPORT void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k,
                            std::complex<double> alpha,
                            cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                            std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                            std::int64_t ldc);

ONEMKL_EXPORT void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                            cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                            cl::sycl::buffer<half, 1> &c, std::int64_t ldc);

} //namespace mklcpu
} //namespace onemkl

#endif //_ONEMKL_BLAS_MKLCPU_HPP_
