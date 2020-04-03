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

//
// Generated based on onemkl/blas/blas.hpp
//

#ifndef _DETAIL_MKLCPU_BLAS_HPP__
#define _DETAIL_MKLCPU_BLAS_HPP__

#include <CL/sycl.hpp>
#include <cstdint>

#include "onemkl/detail/backends.hpp"
#include "onemkl/detail/libraries.hpp"
#include "onemkl/types.hpp"

#include "onemkl_blas_mklcpu.hpp"

namespace onemkl {
namespace blas {

template <onemkl::library lib, onemkl::backend backend>
static inline void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda);
template <>
void syr2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda);
template <>
void syr2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, double alpha,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
template <>
void scal<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    onemkl::mklcpu::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
template <>
void scal<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                double alpha, cl::sycl::buffer<double, 1> &x,
                                                std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    onemkl::mklcpu::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
template <>
void scal<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                std::complex<float> alpha,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    onemkl::mklcpu::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
template <>
void scal<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                std::complex<double> alpha,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    onemkl::mklcpu::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
template <>
void scal<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    onemkl::mklcpu::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
template <>
void scal<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                double alpha,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    onemkl::mklcpu::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
template <>
void trmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
template <>
void trmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);
template <>
void trmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &a,
                                                std::int64_t lda,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx);
template <>
void trmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &a,
                                                std::int64_t lda,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
template <>
void tpmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &a,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
template <>
void tpmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &a,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
template <>
void tpmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &a,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
template <>
void tpmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &a,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a);
template <>
void spr<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, float alpha,
                                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<float, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    onemkl::mklcpu::spr(queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a);
template <>
void spr<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, double alpha,
                                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<double, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    onemkl::mklcpu::spr(queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);
template <>
void hpmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, std::complex<float> alpha,
                                                cl::sycl::buffer<std::complex<float>, 1> &a,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx, std::complex<float> beta,
                                                cl::sycl::buffer<std::complex<float>, 1> &y,
                                                std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    onemkl::mklcpu::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);
template <>
void hpmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, std::complex<double> alpha,
                                                cl::sycl::buffer<std::complex<double>, 1> &a,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx, std::complex<double> beta,
                                                cl::sycl::buffer<std::complex<double>, 1> &y,
                                                std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    onemkl::mklcpu::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c,
                        std::int64_t ldc);
template <>
void syrk<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, std::int64_t n, std::int64_t k,
                                                float alpha, cl::sycl::buffer<float, 1> &a,
                                                std::int64_t lda, float beta,
                                                cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    onemkl::mklcpu::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c,
                        std::int64_t ldc);
template <>
void syrk<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, std::int64_t n, std::int64_t k,
                                                double alpha, cl::sycl::buffer<double, 1> &a,
                                                std::int64_t lda, double beta,
                                                cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    onemkl::mklcpu::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
template <>
void syrk<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    onemkl::mklcpu::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);
template <>
void syrk<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    onemkl::mklcpu::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);
template <>
void her2<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);
template <>
void her2<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
template <>
void hbmv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
template <>
void hbmv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                       float s);
template <>
void rot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                               cl::sycl::buffer<std::complex<float>, 1> &x,
                                               std::int64_t incx,
                                               cl::sycl::buffer<std::complex<float>, 1> &y,
                                               std::int64_t incy, float c, float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    onemkl::mklcpu::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                       double s);
template <>
void rot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                               cl::sycl::buffer<std::complex<double>, 1> &x,
                                               std::int64_t incx,
                                               cl::sycl::buffer<std::complex<double>, 1> &y,
                                               std::int64_t incy, double c, double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    onemkl::mklcpu::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c,
                       float s);
template <>
void rot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                               float c, float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    onemkl::mklcpu::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       double c, double s);
template <>
void rot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                               double c, double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    onemkl::mklcpu::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
template <>
void axpy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    onemkl::mklcpu::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
template <>
void axpy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                double alpha, cl::sycl::buffer<double, 1> &x,
                                                std::int64_t incx, cl::sycl::buffer<double, 1> &y,
                                                std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    onemkl::mklcpu::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
template <>
void axpy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                std::complex<float> alpha,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<float>, 1> &y,
                                                std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    onemkl::mklcpu::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
template <>
void axpy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                std::complex<double> alpha,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<double>, 1> &y,
                                                std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    onemkl::mklcpu::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);
template <>
void gerc<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);
template <>
void gerc<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                         float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
template <>
void syr2k<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                 transpose trans, std::int64_t n, std::int64_t k,
                                                 float alpha, cl::sycl::buffer<float, 1> &a,
                                                 std::int64_t lda, cl::sycl::buffer<float, 1> &b,
                                                 std::int64_t ldb, float beta,
                                                 cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                         double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
template <>
void syr2k<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                 transpose trans, std::int64_t n, std::int64_t k,
                                                 double alpha, cl::sycl::buffer<double, 1> &a,
                                                 std::int64_t lda, cl::sycl::buffer<double, 1> &b,
                                                 std::int64_t ldb, double beta,
                                                 cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                         std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                         std::int64_t ldc);
template <>
void syr2k<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                         std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                         std::int64_t ldc);
template <>
void syr2k<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
template <>
void gemv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, transpose trans,
                                                std::int64_t m, std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                float beta, cl::sycl::buffer<float, 1> &y,
                                                std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
template <>
void gemv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, transpose trans,
                                                std::int64_t m, std::int64_t n, double alpha,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                double beta, cl::sycl::buffer<double, 1> &y,
                                                std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
template <>
void gemv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
template <>
void gemv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
template <>
void her<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, float alpha,
                                               cl::sycl::buffer<std::complex<float>, 1> &x,
                                               std::int64_t incx,
                                               cl::sycl::buffer<std::complex<float>, 1> &a,
                                               std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    onemkl::mklcpu::her(queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
template <>
void her<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, double alpha,
                                               cl::sycl::buffer<std::complex<double>, 1> &x,
                                               std::int64_t incx,
                                               cl::sycl::buffer<std::complex<double>, 1> &a,
                                               std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    onemkl::mklcpu::her(queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a);
template <>
void hpr<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, float alpha,
                                               cl::sycl::buffer<std::complex<float>, 1> &x,
                                               std::int64_t incx,
                                               cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    onemkl::mklcpu::hpr(queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a);
template <>
void hpr<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, double alpha,
                                               cl::sycl::buffer<std::complex<double>, 1> &x,
                                               std::int64_t incx,
                                               cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    onemkl::mklcpu::hpr(queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamin<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamin<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamin<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<std::complex<float>, 1> &x,
                                                 std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamin<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<std::complex<double>, 1> &x,
                                                 std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                        std::int64_t incy);
template <>
void spmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &a,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                float beta, cl::sycl::buffer<float, 1> &y,
                                                std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    onemkl::mklcpu::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                        std::int64_t incy);
template <>
void spmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, double alpha,
                                                cl::sycl::buffer<double, 1> &a,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                double beta, cl::sycl::buffer<double, 1> &y,
                                                std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    onemkl::mklcpu::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
template <>
void swap<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
template <>
void swap<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
template <>
void swap<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<float>, 1> &y,
                                                std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
template <>
void swap<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<double>, 1> &y,
                                                std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda);
template <>
void geru<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda);
template <>
void geru<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);
template <>
void nrm2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);
template <>
void nrm2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result);
template <>
void nrm2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result);
template <>
void nrm2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
template <>
void gemm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, transpose transa,
                                                transpose transb, std::int64_t m, std::int64_t n,
                                                std::int64_t k, float alpha,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                                                float beta, cl::sycl::buffer<float, 1> &c,
                                                std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
template <>
void gemm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, transpose transa,
                                                transpose transb, std::int64_t m, std::int64_t n,
                                                std::int64_t k, double alpha,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                                                double beta, cl::sycl::buffer<double, 1> &c,
                                                std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
template <>
void gemm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
    std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
    std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);
template <>
void gemm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
    std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                        cl::sycl::buffer<half, 1> &c, std::int64_t ldc);
template <>
void gemm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, transpose transa,
                                                transpose transb, std::int64_t m, std::int64_t n,
                                                std::int64_t k, half alpha,
                                                cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<half, 1> &b, std::int64_t ldb,
                                                half beta, cl::sycl::buffer<half, 1> &c,
                                                std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
template <>
void herk<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    onemkl::mklcpu::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);
template <>
void herk<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    onemkl::mklcpu::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);
template <>
void ger<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t m,
                                               std::int64_t n, float alpha,
                                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                               cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);
template <>
void ger<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t m,
                                               std::int64_t n, double alpha,
                                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                               cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    onemkl::mklcpu::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
template <>
void trsm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, side left_right,
                                                uplo upper_lower, transpose trans, diag unit_diag,
                                                std::int64_t m, std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
template <>
void trsm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, side left_right,
                                                uplo upper_lower, transpose trans, diag unit_diag,
                                                std::int64_t m, std::int64_t n, double alpha,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
template <>
void trsm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<float> alpha,
    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
template <>
void trsm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<double> alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);
template <>
void dotu<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<float>, 1> &y,
                                                std::int64_t incy,
                                                cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    onemkl::mklcpu::dotu(queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);
template <>
void dotu<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<double>, 1> &y,
                                                std::int64_t incy,
                                                cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    onemkl::mklcpu::dotu(queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
template <>
void hemm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);
template <>
void hemm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a);
template <>
void hpr2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, std::complex<float> alpha,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<float>, 1> &y,
                                                std::int64_t incy,
                                                cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    onemkl::mklcpu::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a);
template <>
void hpr2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, std::complex<double> alpha,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<double>, 1> &y,
                                                std::int64_t incy,
                                                cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    onemkl::mklcpu::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
template <>
void gbmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, transpose trans,
                                                std::int64_t m, std::int64_t n, std::int64_t kl,
                                                std::int64_t ku, float alpha,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                float beta, cl::sycl::buffer<float, 1> &y,
                                                std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
template <>
void gbmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, transpose trans,
                                                std::int64_t m, std::int64_t n, std::int64_t kl,
                                                std::int64_t ku, double alpha,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                double beta, cl::sycl::buffer<double, 1> &y,
                                                std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy);
template <>
void gbmv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
    std::int64_t ku, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
    std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
    std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy);
template <>
void gbmv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
    std::int64_t ku, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
    std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);
template <>
void tbmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                std::int64_t k, cl::sycl::buffer<float, 1> &a,
                                                std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                                std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);
template <>
void tbmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                std::int64_t k, cl::sycl::buffer<double, 1> &a,
                                                std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                                std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);
template <>
void tbmv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
template <>
void tbmv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
template <>
void symm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, side left_right,
                                                uplo upper_lower, std::int64_t m, std::int64_t n,
                                                float alpha, cl::sycl::buffer<float, 1> &a,
                                                std::int64_t lda, cl::sycl::buffer<float, 1> &b,
                                                std::int64_t ldb, float beta,
                                                cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                        double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
template <>
void symm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, side left_right,
                                                uplo upper_lower, std::int64_t m, std::int64_t n,
                                                double alpha, cl::sycl::buffer<double, 1> &a,
                                                std::int64_t lda, cl::sycl::buffer<double, 1> &b,
                                                std::int64_t ldb, double beta,
                                                cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
template <>
void symm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);
template <>
void symm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result);
template <>
void dotc<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<float>, 1> &y,
                                                std::int64_t incy,
                                                cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    onemkl::mklcpu::dotc(queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result);
template <>
void dotc<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<double>, 1> &y,
                                                std::int64_t incy,
                                                cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    onemkl::mklcpu::dotc(queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);
template <>
void syr<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, float alpha,
                                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    onemkl::mklcpu::syr(queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);
template <>
void syr<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                               std::int64_t n, double alpha,
                                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    onemkl::mklcpu::syr(queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
template <>
void trmm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, side left_right,
                                                uplo upper_lower, transpose trans, diag unit_diag,
                                                std::int64_t m, std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
template <>
void trmm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, side left_right,
                                                uplo upper_lower, transpose trans, diag unit_diag,
                                                std::int64_t m, std::int64_t n, double alpha,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
template <>
void trmm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<float> alpha,
    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
template <>
void trmm<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<double> alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    onemkl::mklcpu::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                         ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                         cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
                         cl::sycl::buffer<float, 1> &param);
template <>
void rotmg<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue,
                                                 cl::sycl::buffer<float, 1> &d1,
                                                 cl::sycl::buffer<float, 1> &d2,
                                                 cl::sycl::buffer<float, 1> &x1, float y1,
                                                 cl::sycl::buffer<float, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    onemkl::mklcpu::rotmg(queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                         cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                         double y1, cl::sycl::buffer<double, 1> &param);
template <>
void rotmg<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue,
                                                 cl::sycl::buffer<double, 1> &d1,
                                                 cl::sycl::buffer<double, 1> &d2,
                                                 cl::sycl::buffer<double, 1> &x1, double y1,
                                                 cl::sycl::buffer<double, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    onemkl::mklcpu::rotmg(queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
template <>
void tpsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &a,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
template <>
void tpsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &a,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
template <>
void tpsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &a,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
template <>
void tpsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &a,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    onemkl::mklcpu::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx);
template <>
void trsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx);
template <>
void trsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);
template <>
void trsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &a,
                                                std::int64_t lda,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx);
template <>
void trsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &a,
                                                std::int64_t lda,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    onemkl::mklcpu::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
template <>
void copy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
template <>
void copy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
template <>
void copy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<float>, 1> &y,
                                                std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
template <>
void copy<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<std::complex<double>, 1> &y,
                                                std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    onemkl::mklcpu::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
template <>
void hemv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
template <>
void hemv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx, std::complex<double> beta,
    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
template <>
void sbmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, std::int64_t k, float alpha,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                float beta, cl::sycl::buffer<float, 1> &y,
                                                std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
template <>
void sbmv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, std::int64_t k, double alpha,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                double beta, cl::sycl::buffer<double, 1> &y,
                                                std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);
template <>
void asum<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<float>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);
template <>
void asum<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<std::complex<double>, 1> &x,
                                                std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result);
template <>
void asum<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result);
template <>
void asum<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);
template <>
void tbsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                std::int64_t k, cl::sycl::buffer<float, 1> &a,
                                                std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                                std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);
template <>
void tbsv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                transpose trans, diag unit_diag, std::int64_t n,
                                                std::int64_t k, cl::sycl::buffer<double, 1> &a,
                                                std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                                std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx);
template <>
void tbsv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
template <>
void tbsv<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    onemkl::mklcpu::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a);
template <>
void spr2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                                cl::sycl::buffer<float, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    onemkl::mklcpu::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a);
template <>
void spr2<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, double alpha,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                                cl::sycl::buffer<double, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    onemkl::mklcpu::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamax<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamax<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamax<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<std::complex<float>, 1> &x,
                                                 std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result);
template <>
void iamax<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                 cl::sycl::buffer<std::complex<double>, 1> &x,
                                                 std::int64_t incx,
                                                 cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    onemkl::mklcpu::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &param);
template <>
void rotm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                                cl::sycl::buffer<float, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    onemkl::mklcpu::rotm(queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &param);
template <>
void rotm<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                                cl::sycl::buffer<double, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    onemkl::mklcpu::rotm(queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<float, 1> &s);
template <>
void rotg<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue,
                                                cl::sycl::buffer<float, 1> &a,
                                                cl::sycl::buffer<float, 1> &b,
                                                cl::sycl::buffer<float, 1> &c,
                                                cl::sycl::buffer<float, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    onemkl::mklcpu::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<double, 1> &s);
template <>
void rotg<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue,
                                                cl::sycl::buffer<double, 1> &a,
                                                cl::sycl::buffer<double, 1> &b,
                                                cl::sycl::buffer<double, 1> &c,
                                                cl::sycl::buffer<double, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    onemkl::mklcpu::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<std::complex<float>, 1> &s);
template <>
void rotg<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue,
                                                cl::sycl::buffer<std::complex<float>, 1> &a,
                                                cl::sycl::buffer<std::complex<float>, 1> &b,
                                                cl::sycl::buffer<float, 1> &c,
                                                cl::sycl::buffer<std::complex<float>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    onemkl::mklcpu::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &b,
                        cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<std::complex<double>, 1> &s);
template <>
void rotg<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue,
                                                cl::sycl::buffer<std::complex<double>, 1> &a,
                                                cl::sycl::buffer<std::complex<double>, 1> &b,
                                                cl::sycl::buffer<double, 1> &c,
                                                cl::sycl::buffer<std::complex<double>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    onemkl::mklcpu::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &result);
template <>
void sdsdot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n, float sb,
                                                  cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                  cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                                  cl::sycl::buffer<float, 1> &result) {
    sdsdot_precondition(queue, n, sb, x, incx, y, incy, result);
    onemkl::mklcpu::sdsdot(queue, n, sb, x, incx, y, incy, result);
    sdsdot_postcondition(queue, n, sb, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
template <>
void her2k<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                         double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                         std::int64_t ldc);
template <>
void her2k<library::intelmkl, backend::intelcpu>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    onemkl::mklcpu::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &result);
template <>
void dot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                               cl::sycl::buffer<float, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    onemkl::mklcpu::dot(queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);
template <>
void dot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                               cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    onemkl::mklcpu::dot(queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);
template <>
void dot<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, std::int64_t n,
                                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                               cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                               cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    onemkl::mklcpu::dot(queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy);
template <>
void symv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, float alpha,
                                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                                float beta, cl::sycl::buffer<float, 1> &y,
                                                std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <onemkl::library lib, onemkl::backend backend>
static inline void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy);
template <>
void symv<library::intelmkl, backend::intelcpu>(cl::sycl::queue &queue, uplo upper_lower,
                                                std::int64_t n, double alpha,
                                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                                double beta, cl::sycl::buffer<double, 1> &y,
                                                std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    onemkl::mklcpu::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

} //namespace blas
} //namespace onemkl

#endif //_DETAIL_MKLCPU_BLAS_HPP_
