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

#ifndef _BLAS_FUNCTION_TABLE_HPP_
#define _BLAS_FUNCTION_TABLE_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>
#include "oneapi/mkl/types.hpp"

typedef struct {
    int version;

    // Buffer APIs

    void (*scasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result);
    void (*dzasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result);
    void (*sasum_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &result);
    void (*dasum_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &result);
    void (*saxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*daxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*caxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*zaxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*scopy_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dcopy_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*ccopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*zcopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*sdot_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                      std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                      cl::sycl::buffer<float, 1> &result);
    void (*ddot_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                      std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                      cl::sycl::buffer<double, 1> &result);
    void (*dsdot_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result);
    void (*cdotc_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<std::complex<float>, 1> &result);
    void (*zdotc_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<std::complex<double>, 1> &result);
    void (*cdotu_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<std::complex<float>, 1> &result);
    void (*zdotu_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<std::complex<double>, 1> &result);
    void (*isamin_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
    void (*idamin_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
    void (*icamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::int64_t, 1> &result);
    void (*izamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::int64_t, 1> &result);
    void (*isamax_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
    void (*idamax_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
    void (*icamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::int64_t, 1> &result);
    void (*izamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::int64_t, 1> &result);
    void (*snrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &result);
    void (*dnrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &result);
    void (*scnrm2_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result);
    void (*dznrm2_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result);
    void (*srot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                      cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                      float s);
    void (*drot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                      cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                      double s);
    void (*csrot_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c,
                       float s);
    void (*zdrot_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       double c, double s);
    void (*srotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                       cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                       cl::sycl::buffer<float, 1> &s);
    void (*drotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                       cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                       cl::sycl::buffer<double, 1> &s);
    void (*crotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                       cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
                       cl::sycl::buffer<std::complex<float>, 1> &s);
    void (*zrotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                       cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
                       cl::sycl::buffer<std::complex<double>, 1> &s);
    void (*srotm_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &param);
    void (*drotm_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &param);
    void (*srotmg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                        cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
                        cl::sycl::buffer<float, 1> &param);
    void (*drotmg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                        cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1, double y1,
                        cl::sycl::buffer<double, 1> &param);
    void (*sscal_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dscal_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*cscal_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*csscal_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*zscal_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*zdscal_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*sdsdot_sycl)(cl::sycl::queue &queue, std::int64_t n, float sb,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &result);
    void (*sswap_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dswap_sycl)(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*cswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*zswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*sgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*cgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy);
    void (*sgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*cgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy);
    void (*sger_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                      cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                      cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*dger_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                      cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                      cl::sycl::buffer<double, 1> &a, std::int64_t lda);
    void (*cgerc_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda);
    void (*zgerc_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda);
    void (*cgeru_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda);
    void (*zgeru_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda);
    void (*chbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zhbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy);
    void (*chemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, std::complex<float> beta,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*zhemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, std::complex<double> beta,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*cher_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*zher_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*cher2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda);
    void (*zher2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda);
    void (*chpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zhpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy);
    void (*chpr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*zhpr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*chpr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*zhpr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*ssbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dsbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*sspmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                       std::int64_t incy);
    void (*dspmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                       std::int64_t incy);
    void (*sspr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<float, 1> &a);
    void (*dspr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<double, 1> &a);
    void (*sspr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a);
    void (*dspr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a);
    void (*ssymv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dsymv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*ssyr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*dsyr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<double, 1> &a, std::int64_t lda);
    void (*ssyr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*dsyr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);
    void (*stbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*stbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*stpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*stpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*strmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*strsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*sgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                       cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                       cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*cgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*hgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                       cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
                       std::int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c, std::int64_t ldc);
    void (*chemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zhemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*cherk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       std::int64_t n, std::int64_t k, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
                       cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zherk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       std::int64_t n, std::int64_t k, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
                       cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*cher2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zher2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*ssymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                       float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dsymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                       double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*csymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zsymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*ssyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c,
                       std::int64_t ldc);
    void (*dsyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c,
                       std::int64_t ldc);
    void (*csyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       std::int64_t n, std::int64_t k, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zsyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                       std::int64_t n, std::int64_t k, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*ssyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dsyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*csyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
    void (*zsyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);
    void (*strmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*dtrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*ctrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*ztrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*strsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*dtrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*ctrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*ztrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*sgemm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, float beta,
                                     cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*dgemm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, double beta,
                                     cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*cgemm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<float> beta,
                                     cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*zgemm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<double> beta,
                                     cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*strsm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size);
    void (*dtrsm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size);
    void (*ctrsm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size);
    void (*ztrsm_batch_strided_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size);
    void (*sgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*cgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                        std::int64_t ldb, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*gemm_f16f16f32_ext_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                    oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                    std::int64_t k, float alpha, cl::sycl::buffer<half, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<half, 1> &b,
                                    std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
                                    std::int64_t ldc);
    void (*gemm_s8u8s32_ext_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                  oneapi::mkl::transpose transb, oneapi::mkl::offset offsetc, std::int64_t m,
                                  std::int64_t n, std::int64_t k, float alpha,
                                  cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
                                  cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo,
                                  float beta, cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
                                  cl::sycl::buffer<int32_t, 1> &co);
    void (*sgemm_ext_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                           cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dgemm_ext_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                           cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*cgemm_ext_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                           std::int64_t ldb, std::complex<float> beta,
                           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zgemm_ext_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                           std::int64_t ldb, std::complex<double> beta,
                           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*hgemm_ext_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           half alpha, cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                           cl::sycl::buffer<half, 1> &c, std::int64_t ldc);

    // USM APIs

    cl::sycl::event (*scasum_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                       const std::complex<float> *x, std::int64_t incx,
                                       float *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dzasum_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                       const std::complex<double> *x, std::int64_t incx,
                                       double *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sasum_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                      std::int64_t incx, float *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dasum_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                      std::int64_t incx, double *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*saxpy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                      const float *x, std::int64_t incx, float *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*daxpy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                      const double *x, std::int64_t incx, double *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*caxpy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zaxpy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*saxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x, std::int64_t *incx,
        float **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*daxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
        std::int64_t *incx, double **y, std::int64_t *incy, std::int64_t group_count,
        std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*caxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
        const std::complex<float> **x, std::int64_t *incx, std::complex<float> **y,
        std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*zaxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
        const std::complex<double> **x, std::int64_t *incx, std::complex<double> **y,
        std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*scopy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                      std::int64_t incx, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dcopy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                      std::int64_t incx, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ccopy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zcopy_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sdot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                     std::int64_t incx, const float *y, std::int64_t incy,
                                     float *result,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ddot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                     std::int64_t incx, const double *y, std::int64_t incy,
                                     double *result,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsdot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                      std::int64_t incx, const float *y, std::int64_t incy,
                                      double *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cdotc_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      const std::complex<float> *y, std::int64_t incy,
                                      std::complex<float> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zdotc_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      const std::complex<double> *y, std::int64_t incy,
                                      std::complex<double> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cdotu_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      const std::complex<float> *y, std::int64_t incy,
                                      std::complex<float> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zdotu_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      const std::complex<double> *y, std::int64_t incy,
                                      std::complex<double> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*isamin_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                       std::int64_t incx, std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*idamin_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                       std::int64_t incx, std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*icamin_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                       const std::complex<float> *x, std::int64_t incx,
                                       std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*izamin_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                       const std::complex<double> *x, std::int64_t incx,
                                       std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*isamax_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                       std::int64_t incx, std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*idamax_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                       std::int64_t incx, std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*icamax_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                       const std::complex<float> *x, std::int64_t incx,
                                       std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*izamax_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                       const std::complex<double> *x, std::int64_t incx,
                                       std::int64_t *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*snrm2_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      float *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dnrm2_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      double *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*scnrm2_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                       std::int64_t incx, float *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dznrm2_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                       std::int64_t incx, double *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*srot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                                     float c, float s,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*drot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     std::complex<double> *x, std::int64_t incx,
                                     std::complex<double> *y, std::int64_t incy, double c, double s,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*csrot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, float *x,
                                      std::int64_t incx, float *y, std::int64_t incy, float c,
                                      float s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zdrot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, double *x,
                                      std::int64_t incx, double *y, std::int64_t incy, double c,
                                      double s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*srotg_usm_sycl)(cl::sycl::queue &queue, float *a, float *b, float *c,
                                      float *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*drotg_usm_sycl)(cl::sycl::queue &queue, double *a, double *b, double *c,
                                      double *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*crotg_usm_sycl)(cl::sycl::queue &queue, std::complex<float> *a,
                                      std::complex<float> *b, float *c, std::complex<float> *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zrotg_usm_sycl)(cl::sycl::queue &queue, std::complex<double> *a,
                                      std::complex<double> *b, double *c, std::complex<double> *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*srotm_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, float *x,
                                      std::int64_t incx, float *y, std::int64_t incy, float *param,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*drotm_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, double *x,
                                      std::int64_t incx, double *y, std::int64_t incy,
                                      double *param,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*srotmg_usm_sycl)(cl::sycl::queue &queue, float *d1, float *d2, float *x1,
                                       float y1, float *param,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*drotmg_usm_sycl)(cl::sycl::queue &queue, double *d1, double *d2, double *x1,
                                       double y1, double *param,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sscal_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dscal_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                      double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cscal_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<float> alpha, std::complex<float> *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*csscal_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                       std::complex<double> alpha, std::complex<double> *x,
                                       std::int64_t incx,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zscal_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zdscal_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                       std::complex<double> *x, std::int64_t incx,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sdsdot_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, float sb,
                                       const float *x, std::int64_t incx, const float *y,
                                       std::int64_t incy, float *result,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sswap_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, float *x,
                                      std::int64_t incx, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dswap_usm_sycl)(cl::sycl::queue &queue, std::int64_t n, double *x,
                                      std::int64_t incx, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cswap_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zswap_usm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sgbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, std::int64_t kl,
                                      std::int64_t ku, float alpha, const float *a,
                                      std::int64_t lda, const float *x, std::int64_t incx,
                                      float beta, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dgbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, std::int64_t kl,
                                      std::int64_t ku, double alpha, const double *a,
                                      std::int64_t lda, const double *x, std::int64_t incx,
                                      double beta, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, std::int64_t kl,
                                      std::int64_t ku, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> beta, std::complex<float> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, std::int64_t kl,
                                      std::int64_t ku, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> beta, std::complex<double> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sgemv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, float alpha, const float *a,
                                      std::int64_t lda, const float *x, std::int64_t incx,
                                      float beta, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dgemv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, double alpha, const double *a,
                                      std::int64_t lda, const double *x, std::int64_t incx,
                                      double beta, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgemv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> beta, std::complex<float> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgemv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                      std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> beta, std::complex<double> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sger_usm_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                     float alpha, const float *x, std::int64_t incx, const float *y,
                                     std::int64_t incy, float *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dger_usm_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                     double alpha, const double *x, std::int64_t incx,
                                     const double *y, std::int64_t incy, double *a,
                                     std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgerc_usm_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, const std::complex<float> *y,
                                      std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgerc_usm_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, const std::complex<double> *y,
                                      std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgeru_usm_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, const std::complex<float> *y,
                                      std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgeru_usm_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, const std::complex<double> *y,
                                      std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*chbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> beta, std::complex<float> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zhbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> beta, std::complex<double> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*chemv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> beta, std::complex<float> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zhemv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> beta, std::complex<double> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cher_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, float alpha, const std::complex<float> *x,
                                     std::int64_t incx, std::complex<float> *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zher_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, double alpha, const std::complex<double> *x,
                                     std::int64_t incx, std::complex<double> *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cher2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *x, std::int64_t incx,
                                      const std::complex<float> *y, std::int64_t incy,
                                      std::complex<float> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zher2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *x, std::int64_t incx,
                                      const std::complex<double> *y, std::int64_t incy,
                                      std::complex<double> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*chpmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, const std::complex<float> *x,
                                      std::int64_t incx, std::complex<float> beta,
                                      std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zhpmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, const std::complex<double> *x,
                                      std::int64_t incx, std::complex<double> beta,
                                      std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*chpr_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, float alpha, const std::complex<float> *x,
                                     std::int64_t incx, std::complex<float> *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zhpr_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, double alpha, const std::complex<double> *x,
                                     std::int64_t incx, std::complex<double> *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*chpr2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *x, std::int64_t incx,
                                      const std::complex<float> *y, std::int64_t incy,
                                      std::complex<float> *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zhpr2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *x, std::int64_t incx,
                                      const std::complex<double> *y, std::int64_t incy,
                                      std::complex<double> *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ssbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::int64_t k, float alpha, const float *a,
                                      std::int64_t lda, const float *x, std::int64_t incx,
                                      float beta, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                                      std::int64_t lda, const double *x, std::int64_t incx,
                                      double beta, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sspmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, float alpha, const float *a, const float *x,
                                      std::int64_t incx, float beta, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dspmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, double alpha, const double *a,
                                      const double *x, std::int64_t incx, double beta, double *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sspr_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, float alpha, const float *x, std::int64_t incx,
                                     float *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dspr_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, double alpha, const double *x,
                                     std::int64_t incx, double *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sspr2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, float alpha, const float *x,
                                      std::int64_t incx, const float *y, std::int64_t incy,
                                      float *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dspr2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, double alpha, const double *x,
                                      std::int64_t incx, const double *y, std::int64_t incy,
                                      double *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ssymv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, float alpha, const float *a, std::int64_t lda,
                                      const float *x, std::int64_t incx, float beta, float *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsymv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, double alpha, const double *a,
                                      std::int64_t lda, const double *x, std::int64_t incx,
                                      double beta, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ssyr_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, float alpha, const float *x, std::int64_t incx,
                                     float *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsyr_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     std::int64_t n, double alpha, const double *x,
                                     std::int64_t incx, double *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ssyr2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, float alpha, const float *x,
                                      std::int64_t incx, const float *y, std::int64_t incy,
                                      float *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsyr2_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      std::int64_t n, double alpha, const double *x,
                                      std::int64_t incx, const double *y, std::int64_t incy,
                                      double *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*stbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const float *a,
                                      std::int64_t lda, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const double *a,
                                      std::int64_t lda, double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztbmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*stbsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const float *a,
                                      std::int64_t lda, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtbsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const double *a,
                                      std::int64_t lda, double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctbsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztbsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, std::int64_t k, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*stpmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const float *a, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtpmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const double *a, double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctpmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<float> *a,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztpmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<double> *a,
                                      std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*stpsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const float *a, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtpsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const double *a, double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctpsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<float> *a,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztpsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<double> *a,
                                      std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*strmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const float *a, std::int64_t lda, float *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtrmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const double *a, std::int64_t lda, double *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctrmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztrmv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*strsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const float *a, std::int64_t lda, float *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtrsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const double *a, std::int64_t lda, double *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctrsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztrsv_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                      std::int64_t n, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sgemm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                      oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                      std::int64_t k, float alpha, const float *a, std::int64_t lda,
                                      const float *b, std::int64_t ldb, float beta, float *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dgemm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                      oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                      std::int64_t k, double alpha, const double *a,
                                      std::int64_t lda, const double *b, std::int64_t ldb,
                                      double beta, double *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgemm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                      oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                      std::int64_t k, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *b, std::int64_t ldb,
                                      std::complex<float> beta, std::complex<float> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgemm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                      oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                      std::int64_t k, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *b, std::int64_t ldb,
                                      std::complex<double> beta, std::complex<double> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*chemm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, const std::complex<float> *b,
                                      std::int64_t ldb, std::complex<float> beta,
                                      std::complex<float> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zhemm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, const std::complex<double> *b,
                                      std::int64_t ldb, std::complex<double> beta,
                                      std::complex<double> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cherk_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                      float alpha, const std::complex<float> *a, std::int64_t lda,
                                      float beta, std::complex<float> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zherk_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                      double alpha, const std::complex<double> *a, std::int64_t lda,
                                      double beta, std::complex<double> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cher2k_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                       std::complex<float> alpha, const std::complex<float> *a,
                                       std::int64_t lda, const std::complex<float> *b,
                                       std::int64_t ldb, float beta, std::complex<float> *c,
                                       std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zher2k_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                       std::complex<double> alpha, const std::complex<double> *a,
                                       std::int64_t lda, const std::complex<double> *b,
                                       std::int64_t ldb, double beta, std::complex<double> *c,
                                       std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ssymm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                      float alpha, const float *a, std::int64_t lda, const float *b,
                                      std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsymm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                      double alpha, const double *a, std::int64_t lda,
                                      const double *b, std::int64_t ldb, double beta, double *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*csymm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, const std::complex<float> *b,
                                      std::int64_t ldb, std::complex<float> beta,
                                      std::complex<float> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zsymm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, const std::complex<double> *b,
                                      std::int64_t ldb, std::complex<double> beta,
                                      std::complex<double> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ssyrk_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                      float alpha, const float *a, std::int64_t lda, float beta,
                                      float *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsyrk_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                      double alpha, const double *a, std::int64_t lda, double beta,
                                      double *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*csyrk_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> beta,
                                      std::complex<float> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zsyrk_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> beta,
                                      std::complex<double> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ssyr2k_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                       float alpha, const float *a, std::int64_t lda,
                                       const float *b, std::int64_t ldb, float beta, float *c,
                                       std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dsyr2k_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                       double alpha, const double *a, std::int64_t lda,
                                       const double *b, std::int64_t ldb, double beta, double *c,
                                       std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*csyr2k_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                       std::complex<float> alpha, const std::complex<float> *a,
                                       std::int64_t lda, const std::complex<float> *b,
                                       std::int64_t ldb, std::complex<float> beta,
                                       std::complex<float> *c, std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zsyr2k_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                       std::complex<double> alpha, const std::complex<double> *a,
                                       std::int64_t lda, const std::complex<double> *b,
                                       std::int64_t ldb, std::complex<double> beta,
                                       std::complex<double> *c, std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*strmm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      float alpha, const float *a, std::int64_t lda, float *b,
                                      std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtrmm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      double alpha, const double *a, std::int64_t lda, double *b,
                                      std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctrmm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztrmm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*strsm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      float alpha, const float *a, std::int64_t lda, float *b,
                                      std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dtrsm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      double alpha, const double *a, std::int64_t lda, double *b,
                                      std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ctrsm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*ztrsm_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                      oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                      oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha, const float **a,
        std::int64_t *lda, const float **b, std::int64_t *ldb, float *beta, float **c,
        std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha, const double **a,
        std::int64_t *lda, const double **b, std::int64_t *ldb, double *beta, double **c,
        std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
        const std::complex<float> **a, std::int64_t *lda, const std::complex<float> **b,
        std::int64_t *ldb, std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
        std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
        const std::complex<double> **a, std::int64_t *lda, const std::complex<double> **b,
        std::int64_t *ldb, std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
        std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t m,
        std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
        std::int64_t stride_a, const float *b, std::int64_t ldb, std::int64_t stride_b, float beta,
        float *c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t m,
        std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
        std::int64_t stride_a, const double *b, std::int64_t ldb, std::int64_t stride_b,
        double beta, double *c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t m,
        std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b, std::int64_t ldb,
        std::int64_t stride_b, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t m,
        std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b, std::int64_t ldb,
        std::int64_t stride_b, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*sgemmt_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                       std::int64_t n, std::int64_t k, float alpha, const float *a,
                                       std::int64_t lda, const float *b, std::int64_t ldb,
                                       float beta, float *c, std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*dgemmt_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                       std::int64_t n, std::int64_t k, double alpha,
                                       const double *a, std::int64_t lda, const double *b,
                                       std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*cgemmt_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                       std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                       const std::complex<float> *a, std::int64_t lda,
                                       const std::complex<float> *b, std::int64_t ldb,
                                       std::complex<float> beta, std::complex<float> *c,
                                       std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*zgemmt_usm_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                       oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                       std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                       const std::complex<double> *a, std::int64_t lda,
                                       const std::complex<double> *b, std::int64_t ldb,
                                       std::complex<double> beta, std::complex<double> *c,
                                       std::int64_t ldc,
                                       const cl::sycl::vector_class<cl::sycl::event> &dependencies);

} function_table_t;

#endif //_BLAS_FUNCTION_TABLE_HPP_
