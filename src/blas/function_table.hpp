/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <complex>
#include <cstdint>
#include <CL/sycl.hpp>
#include "oneapi/mkl/types.hpp"

typedef struct {
    int version;

    // Buffer APIs

    void (*column_major_scasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<float, 1> &result);
    void (*column_major_dzasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx, cl::sycl::buffer<double, 1> &result);
    void (*column_major_sasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &result);
    void (*column_major_dasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<double, 1> &result);
    void (*column_major_saxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*column_major_daxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*column_major_caxpy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zaxpy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_scopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*column_major_dcopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*column_major_ccopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zcopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_sdot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                   cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                   cl::sycl::buffer<float, 1> &result);
    void (*column_major_ddot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                   cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                   cl::sycl::buffer<double, 1> &result);
    void (*column_major_dsdot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<double, 1> &result);
    void (*column_major_cdotc_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<float>, 1> &result);
    void (*column_major_zdotc_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<double>, 1> &result);
    void (*column_major_cdotu_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<float>, 1> &result);
    void (*column_major_zdotu_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<double>, 1> &result);
    void (*column_major_isamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_idamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_icamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_izamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_isamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_idamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_icamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_izamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
    void (*column_major_scnrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<float, 1> &result);
    void (*column_major_dznrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     cl::sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx, cl::sycl::buffer<double, 1> &result);
    void (*column_major_snrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &result);
    void (*column_major_dnrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<double, 1> &result);
    void (*column_major_srot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                   cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                   float c, float s);
    void (*column_major_drot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                   cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                   double c, double s);
    void (*column_major_csrot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c,
                                    float s);
    void (*column_major_zdrot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c,
                                    double s);
    void (*column_major_srotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                                    cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                                    cl::sycl::buffer<float, 1> &s);
    void (*column_major_drotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                                    cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                                    cl::sycl::buffer<double, 1> &s);
    void (*column_major_crotg_sycl)(cl::sycl::queue &queue,
                                    cl::sycl::buffer<std::complex<float>, 1> &a,
                                    cl::sycl::buffer<std::complex<float>, 1> &b,
                                    cl::sycl::buffer<float, 1> &c,
                                    cl::sycl::buffer<std::complex<float>, 1> &s);
    void (*column_major_zrotg_sycl)(cl::sycl::queue &queue,
                                    cl::sycl::buffer<std::complex<double>, 1> &a,
                                    cl::sycl::buffer<std::complex<double>, 1> &b,
                                    cl::sycl::buffer<double, 1> &c,
                                    cl::sycl::buffer<std::complex<double>, 1> &s);
    void (*column_major_srotm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<float, 1> &param);
    void (*column_major_drotm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<double, 1> &param);
    void (*column_major_srotmg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                                     cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1,
                                     float y1, cl::sycl::buffer<float, 1> &param);
    void (*column_major_drotmg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                                     cl::sycl::buffer<double, 1> &d2,
                                     cl::sycl::buffer<double, 1> &x1, double y1,
                                     cl::sycl::buffer<double, 1> &param);
    void (*column_major_sscal_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*column_major_dscal_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*column_major_cscal_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*column_major_csscal_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                     std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx);
    void (*column_major_zscal_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*column_major_zdscal_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx);
    void (*column_major_sdsdot_sycl)(cl::sycl::queue &queue, std::int64_t n, float sb,
                                     cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                     cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                     cl::sycl::buffer<float, 1> &result);
    void (*column_major_sswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*column_major_dswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*column_major_cswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_sgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, std::int64_t kl,
                                    std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                    std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                                    std::int64_t incy);
    void (*column_major_dgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, std::int64_t kl,
                                    std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                                    std::int64_t incy);
    void (*column_major_cgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, std::int64_t kl,
                                    std::int64_t ku, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, std::int64_t kl,
                                    std::int64_t ku, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_sgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, float alpha,
                                    cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*column_major_dgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, double alpha,
                                    cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*column_major_cgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                    std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_sger_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                   float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                   cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*column_major_dger_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                   double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                   cl::sycl::buffer<double, 1> &a, std::int64_t lda);
    void (*column_major_cgerc_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*column_major_zgerc_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*column_major_cgeru_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*column_major_zgeru_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*column_major_chbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zhbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_chemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zhemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_cher_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, float alpha,
                                   cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*column_major_zher_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, double alpha,
                                   cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*column_major_cher2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*column_major_zher2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*column_major_chpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*column_major_zhpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &y,
                                    std::int64_t incy);
    void (*column_major_chpr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, float alpha,
                                   cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*column_major_zhpr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, double alpha,
                                   cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                   cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*column_major_chpr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*column_major_zhpr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*column_major_ssbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::int64_t k, float alpha,
                                    cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*column_major_dsbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, std::int64_t k, double alpha,
                                    cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*column_major_sspmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                    cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*column_major_dspmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                    cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*column_major_sspr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                   std::int64_t incx, cl::sycl::buffer<float, 1> &a);
    void (*column_major_dspr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                   std::int64_t incx, cl::sycl::buffer<double, 1> &a);
    void (*column_major_sspr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                    std::int64_t incx, cl::sycl::buffer<float, 1> &y,
                                    std::int64_t incy, cl::sycl::buffer<float, 1> &a);
    void (*column_major_dspr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx, cl::sycl::buffer<double, 1> &y,
                                    std::int64_t incy, cl::sycl::buffer<double, 1> &a);
    void (*column_major_ssymv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                    std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                                    std::int64_t incy);
    void (*column_major_dsymv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                                    std::int64_t incy);
    void (*column_major_ssyr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                   std::int64_t incx, cl::sycl::buffer<float, 1> &a,
                                   std::int64_t lda);
    void (*column_major_dsyr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                   std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                   std::int64_t incx, cl::sycl::buffer<double, 1> &a,
                                   std::int64_t lda);
    void (*column_major_ssyr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                    std::int64_t incx, cl::sycl::buffer<float, 1> &y,
                                    std::int64_t incy, cl::sycl::buffer<float, 1> &a,
                                    std::int64_t lda);
    void (*column_major_dsyr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx, cl::sycl::buffer<double, 1> &y,
                                    std::int64_t incy, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda);
    void (*column_major_stbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                    std::int64_t incx);
    void (*column_major_dtbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx);
    void (*column_major_ctbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*column_major_ztbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_stbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                    std::int64_t incx);
    void (*column_major_dtbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx);
    void (*column_major_ctbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*column_major_ztbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, std::int64_t k,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_stpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<float, 1> &a,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*column_major_dtpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<double, 1> &a,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*column_major_ctpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*column_major_ztpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                    cl::sycl::buffer<std::complex<double>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_stpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<float, 1> &a,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*column_major_dtpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<double, 1> &a,
                                    cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*column_major_ctpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                    cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*column_major_ztpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                    cl::sycl::buffer<std::complex<double>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_strmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*column_major_dtrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx);
    void (*column_major_ctrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_ztrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_strsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*column_major_dtrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                    std::int64_t incx);
    void (*column_major_ctrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_ztrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                    std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                                    std::int64_t incx);
    void (*column_major_sgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                    oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                    std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<float, 1> &b,
                                    std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
                                    std::int64_t ldc);
    void (*column_major_dgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                    oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                    std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<double, 1> &b,
                                    std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
                                    std::int64_t ldc);
    void (*column_major_cgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                    oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                    std::int64_t k, std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*column_major_zgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                    oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                    std::int64_t k, std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*column_major_hgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                    oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                    std::int64_t k, cl::sycl::half alpha,
                                    cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb,
                                    cl::sycl::half beta, cl::sycl::buffer<cl::sycl::half, 1> &c,
                                    std::int64_t ldc);
    void (*column_major_gemm_f16f16f32_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                             oneapi::mkl::transpose transb, std::int64_t m,
                                             std::int64_t n, std::int64_t k, float alpha,
                                             cl::sycl::buffer<cl::sycl::half, 1> &a,
                                             std::int64_t lda,
                                             cl::sycl::buffer<cl::sycl::half, 1> &b,
                                             std::int64_t ldb, float beta,
                                             cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*column_major_chemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*column_major_zhemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*column_major_cherk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                    float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                                    std::int64_t lda, float beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*column_major_zherk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                    double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                                    std::int64_t lda, double beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*column_major_cher2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                     std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                                     std::int64_t ldc);
    void (*column_major_zher2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                     std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                                     std::int64_t ldc);
    void (*column_major_ssymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                    float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                    cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*column_major_dsymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                    double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                    cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*column_major_csymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*column_major_zsymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*column_major_ssyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                    float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*column_major_dsyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                    double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*column_major_csyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*column_major_zsyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                    oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*column_major_ssyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                     float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                     cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*column_major_dsyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                     double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                     cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*column_major_csyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                     std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::complex<float> beta,
                                     cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*column_major_zsyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                     std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::complex<double> beta,
                                     cl::sycl::buffer<std::complex<double>, 1> &c,
                                     std::int64_t ldc);
    void (*column_major_strmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*column_major_dtrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*column_major_ctrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*column_major_ztrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*column_major_strsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*column_major_dtrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*column_major_ctrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*column_major_ztrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                    oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                    oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*column_major_sgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
        std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
        std::int64_t stride_b, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size);
    void (*column_major_dgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
        cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b, double beta,
        cl::sycl::buffer<double, 1> &c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size);
    void (*column_major_cgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size);
    void (*column_major_zgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size);
    void (*column_major_strsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::int64_t batch_size);
    void (*column_major_dtrsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::int64_t batch_size);
    void (*column_major_ctrsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
        std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    void (*column_major_ztrsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
        std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    void (*column_major_sgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                     std::int64_t n, std::int64_t k, float alpha,
                                     cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                     cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*column_major_dgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                     std::int64_t n, std::int64_t k, double alpha,
                                     cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                     cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*column_major_cgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::complex<float> beta,
                                     cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*column_major_zgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                     oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::complex<double> beta,
                                     cl::sycl::buffer<std::complex<double>, 1> &c,
                                     std::int64_t ldc);
    void (*column_major_gemm_s8u8s32_bias_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
        cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
        cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
        cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);

    // USM APIs

    cl::sycl::event (*column_major_scasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dzasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_saxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x, std::int64_t incx,
        float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_daxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x, std::int64_t incx,
        double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_caxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zaxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*column_major_saxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x, std::int64_t *incx,
        float **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*column_major_daxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
        std::int64_t *incx, double **y, std::int64_t *incy, std::int64_t group_count,
        std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*column_major_caxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
        const std::complex<float> **x, std::int64_t *incx, std::complex<float> **y,
        std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*column_major_zaxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
        const std::complex<double> **x, std::int64_t *incx, std::complex<double> **y,
        std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*column_major_scopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dcopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ccopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zcopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sdot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y,
        std::int64_t incy, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ddot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, const double *y,
        std::int64_t incy, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsdot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y,
        std::int64_t incy, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cdotc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zdotc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cdotu_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zdotu_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_isamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_idamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_icamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_izamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_isamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_idamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_icamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_izamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_scnrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dznrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_snrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dnrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_srot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
        std::complex<float> *y, std::int64_t incy, float c, float s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_drot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
        std::complex<double> *y, std::int64_t incy, double c, double s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_csrot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
        std::int64_t incy, float c, float s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zdrot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
        std::int64_t incy, double c, double s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_srotg_usm_sycl)(
        cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_drotg_usm_sycl)(
        cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_crotg_usm_sycl)(
        cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b, float *c,
        std::complex<float> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zrotg_usm_sycl)(
        cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b, double *c,
        std::complex<double> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_srotm_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
        std::int64_t incy, float *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_drotm_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
        std::int64_t incy, double *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_srotmg_usm_sycl)(
        cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1, float *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_drotmg_usm_sycl)(
        cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1, double *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float alpha, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double alpha, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, std::complex<float> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_csscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha, std::complex<double> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zdscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sdsdot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx,
        const float *y, std::int64_t incy, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, float alpha, const float *a, std::int64_t lda,
        const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, double alpha, const double *a, std::int64_t lda,
        const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
        std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
        float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        double alpha, const double *a, std::int64_t lda, const double *x, std::int64_t incx,
        double beta, double *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sger_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha, const float *x,
        std::int64_t incx, const float *y, std::int64_t incy, float *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dger_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha, const double *x,
        std::int64_t incx, const double *y, std::int64_t incy, double *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgerc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
        const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
        std::int64_t incy, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgerc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
        const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
        std::int64_t incy, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgeru_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
        const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
        std::int64_t incy, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgeru_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
        const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
        std::int64_t incy, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_chbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zhbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_chemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zhemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cher_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zher_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cher2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zher2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_chpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, const std::complex<float> *x,
        std::int64_t incx, std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zhpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, const std::complex<double> *x,
        std::int64_t incx, std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_chpr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zhpr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_chpr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zhpr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ssbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
        float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        double alpha, const double *a, std::int64_t lda, const double *x, std::int64_t incx,
        double beta, double *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sspmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *a, const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dspmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *a, const double *x, std::int64_t incx, double beta, double *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sspr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, float *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dspr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, double *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sspr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dspr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ssymv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsymv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *a, std::int64_t lda, const double *x, std::int64_t incx, double beta,
        double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ssyr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, float *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsyr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, double *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ssyr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
        std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsyr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
        std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_stbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
        std::int64_t lda, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
        std::int64_t lda, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a,
        std::int64_t lda, std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_stbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
        std::int64_t lda, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
        std::int64_t lda, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a,
        std::int64_t lda, std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_stpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_stpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_strmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtrmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctrmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztrmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_strsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtrsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctrsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztrsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a,
        std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a,
        std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
        std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
        std::int64_t ldb, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
        std::int64_t ldb, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_chemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zhemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cherk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, float alpha, const std::complex<float> *a, std::int64_t lda,
        float beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zherk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, double alpha, const std::complex<double> *a,
        std::int64_t lda, double beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cher2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, float beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zher2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, double beta,
        std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ssymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
        const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda,
        const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_csymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zsymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ssyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda, float beta,
        float *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
        double beta, double *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_csyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zsyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ssyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
        const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dsyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
        const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_csyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zsyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_strmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        float alpha, const float *a, std::int64_t lda, float *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtrmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        double alpha, const double *a, std::int64_t lda, double *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctrmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztrmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        std::complex<double> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_strsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        float alpha, const float *a, std::int64_t lda, float *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dtrsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        double alpha, const double *a, std::int64_t lda, double *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ctrsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_ztrsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        std::complex<double> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha, const float **a,
        std::int64_t *lda, const float **b, std::int64_t *ldb, float *beta, float **c,
        std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha, const double **a,
        std::int64_t *lda, const double **b, std::int64_t *ldb, double *beta, double **c,
        std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
        const std::complex<float> **a, std::int64_t *lda, const std::complex<float> **b,
        std::int64_t *ldb, std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
        std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
        const std::complex<double> **a, std::int64_t *lda, const std::complex<double> **b,
        std::int64_t *ldb, std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
        std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a,
        std::int64_t lda, std::int64_t stride_a, const float *b, std::int64_t ldb,
        std::int64_t stride_b, float beta, float *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a,
        std::int64_t lda, std::int64_t stride_a, const double *b, std::int64_t ldb,
        std::int64_t stride_b, double beta, double *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        const std::complex<float> *a, std::int64_t lda, std::int64_t stride_a,
        const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<float> beta, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
        const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_sgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha, const float *a,
        std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_dgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
        const double *a, std::int64_t lda, const double *b, std::int64_t ldb, double beta,
        double *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_cgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
        std::int64_t ldb, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*column_major_zgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
        std::int64_t ldb, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    // Buffer APIs

    void (*row_major_scasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<float, 1> &result);
    void (*row_major_dzasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<double, 1> &result);
    void (*row_major_sasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &result);
    void (*row_major_dasum_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<double, 1> &result);
    void (*row_major_saxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_daxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*row_major_caxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zaxpy_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_scopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_dcopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*row_major_ccopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zcopy_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_sdot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                cl::sycl::buffer<float, 1> &result);
    void (*row_major_ddot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                cl::sycl::buffer<double, 1> &result);
    void (*row_major_dsdot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<double, 1> &result);
    void (*row_major_cdotc_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<float>, 1> &result);
    void (*row_major_zdotc_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<double>, 1> &result);
    void (*row_major_cdotu_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<float>, 1> &result);
    void (*row_major_zdotu_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<double>, 1> &result);
    void (*row_major_isamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_idamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_icamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_izamin_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_isamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_idamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_icamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_izamax_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<std::int64_t, 1> &result);
    void (*row_major_scnrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<float, 1> &result);
    void (*row_major_dznrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<double, 1> &result);
    void (*row_major_snrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &result);
    void (*row_major_dnrm2_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<double, 1> &result);
    void (*row_major_srot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                float c, float s);
    void (*row_major_drot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                double c, double s);
    void (*row_major_csrot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c,
                                 float s);
    void (*row_major_zdrot_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c,
                                 double s);
    void (*row_major_srotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                                 cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                                 cl::sycl::buffer<float, 1> &s);
    void (*row_major_drotg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                                 cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                                 cl::sycl::buffer<double, 1> &s);
    void (*row_major_crotg_sycl)(cl::sycl::queue &queue,
                                 cl::sycl::buffer<std::complex<float>, 1> &a,
                                 cl::sycl::buffer<std::complex<float>, 1> &b,
                                 cl::sycl::buffer<float, 1> &c,
                                 cl::sycl::buffer<std::complex<float>, 1> &s);
    void (*row_major_zrotg_sycl)(cl::sycl::queue &queue,
                                 cl::sycl::buffer<std::complex<double>, 1> &a,
                                 cl::sycl::buffer<std::complex<double>, 1> &b,
                                 cl::sycl::buffer<double, 1> &c,
                                 cl::sycl::buffer<std::complex<double>, 1> &s);
    void (*row_major_srotm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<float, 1> &param);
    void (*row_major_drotm_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<double, 1> &param);
    void (*row_major_srotmg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                                  cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1,
                                  float y1, cl::sycl::buffer<float, 1> &param);
    void (*row_major_drotmg_sycl)(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                                  cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                                  double y1, cl::sycl::buffer<double, 1> &param);
    void (*row_major_sscal_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*row_major_dscal_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*row_major_cscal_sycl)(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*row_major_csscal_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                  std::complex<double> alpha,
                                  cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*row_major_zscal_sycl)(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*row_major_zdscal_sycl)(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                  cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*row_major_sdsdot_sycl)(cl::sycl::queue &queue, std::int64_t n, float sb,
                                  cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                  cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                  cl::sycl::buffer<float, 1> &result);
    void (*row_major_sswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_dswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*row_major_cswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zswap_sycl)(cl::sycl::queue &queue, std::int64_t n,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_sgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                 float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_dgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                 double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*row_major_cgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zgbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_sgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, float alpha,
                                 cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_dgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, double alpha,
                                 cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*row_major_cgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zgemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose trans,
                                 std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_sger_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*row_major_dger_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                                cl::sycl::buffer<double, 1> &a, std::int64_t lda);
    void (*row_major_cgerc_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*row_major_zgerc_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*row_major_cgeru_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*row_major_zgeru_sycl)(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*row_major_chbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zhbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_chemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zhemv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_cher_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, float alpha,
                                cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*row_major_zher_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, double alpha,
                                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*row_major_cher2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*row_major_zher2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*row_major_chpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*row_major_zhpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*row_major_chpr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, float alpha,
                                cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*row_major_zhpr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, double alpha,
                                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*row_major_chpr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*row_major_zhpr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*row_major_ssbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::int64_t k, float alpha,
                                 cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_dsbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, std::int64_t k, double alpha,
                                 cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*row_major_sspmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_dspmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                 cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*row_major_sspr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                std::int64_t incx, cl::sycl::buffer<float, 1> &a);
    void (*row_major_dspr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                std::int64_t incx, cl::sycl::buffer<double, 1> &a);
    void (*row_major_sspr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                 std::int64_t incx, cl::sycl::buffer<float, 1> &y,
                                 std::int64_t incy, cl::sycl::buffer<float, 1> &a);
    void (*row_major_dspr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                 std::int64_t incx, cl::sycl::buffer<double, 1> &y,
                                 std::int64_t incy, cl::sycl::buffer<double, 1> &a);
    void (*row_major_ssymv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*row_major_dsymv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                 std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                                 std::int64_t incy);
    void (*row_major_ssyr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                std::int64_t incx, cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*row_major_dsyr_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                std::int64_t incx, cl::sycl::buffer<double, 1> &a,
                                std::int64_t lda);
    void (*row_major_ssyr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                                 std::int64_t incx, cl::sycl::buffer<float, 1> &y,
                                 std::int64_t incy, cl::sycl::buffer<float, 1> &a,
                                 std::int64_t lda);
    void (*row_major_dsyr2_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
                                 std::int64_t incx, cl::sycl::buffer<double, 1> &y,
                                 std::int64_t incy, cl::sycl::buffer<double, 1> &a,
                                 std::int64_t lda);
    void (*row_major_stbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                 std::int64_t incx);
    void (*row_major_dtbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                 std::int64_t incx);
    void (*row_major_ctbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*row_major_ztbmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*row_major_stbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<float, 1> &x,
                                 std::int64_t incx);
    void (*row_major_dtbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                                 std::int64_t incx);
    void (*row_major_ctbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*row_major_ztbsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, std::int64_t k,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*row_major_stpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<float, 1> &a,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*row_major_dtpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<double, 1> &a,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*row_major_ctpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*row_major_ztpmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*row_major_stpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<float, 1> &a,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*row_major_dtpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<double, 1> &a,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*row_major_ctpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                 cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*row_major_ztpsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                 cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*row_major_strmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*row_major_dtrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*row_major_ctrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                                 std::int64_t incx);
    void (*row_major_ztrmv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                                 std::int64_t incx);
    void (*row_major_strsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*row_major_dtrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*row_major_ctrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                                 std::int64_t incx);
    void (*row_major_ztrsv_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag,
                                 std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                                 std::int64_t incx);
    void (*row_major_sgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                 oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                 std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                                 float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*row_major_dgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                 oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                 std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                                 std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                                 double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*row_major_cgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                 oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                 std::int64_t k, std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*row_major_zgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                 oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                 std::int64_t k, std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*row_major_hgemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                 oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                                 std::int64_t k, cl::sycl::half alpha,
                                 cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb,
                                 cl::sycl::half beta, cl::sycl::buffer<cl::sycl::half, 1> &c,
                                 std::int64_t ldc);
    void (*row_major_gemm_f16f16f32_sycl)(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                                          oneapi::mkl::transpose transb, std::int64_t m,
                                          std::int64_t n, std::int64_t k, float alpha,
                                          cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
                                          cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb,
                                          float beta, cl::sycl::buffer<float, 1> &c,
                                          std::int64_t ldc);
    void (*row_major_chemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*row_major_zhemm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*row_major_cherk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                 float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                                 std::int64_t lda, float beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*row_major_zherk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                 double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                                 std::int64_t lda, double beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*row_major_cher2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                  std::complex<float> alpha,
                                  cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                  float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                                  std::int64_t ldc);
    void (*row_major_zher2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                  std::complex<double> alpha,
                                  cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                  double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                                  std::int64_t ldc);
    void (*row_major_ssymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                 float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                 cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*row_major_dsymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                 double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                 cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*row_major_csymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*row_major_zsymm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*row_major_ssyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                 float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*row_major_dsyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                 double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*row_major_csyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*row_major_zsyrk_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*row_major_ssyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                  float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                  cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*row_major_dsyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                  double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                  cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*row_major_csyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                  std::complex<float> alpha,
                                  cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                  std::complex<float> beta,
                                  cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*row_major_zsyr2k_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                                  std::complex<double> alpha,
                                  cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                  std::complex<double> beta,
                                  cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*row_major_strmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*row_major_dtrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*row_major_ctrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*row_major_ztrmm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*row_major_strsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*row_major_dtrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*row_major_ctrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*row_major_ztrsm_sycl)(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*row_major_sgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
        std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
        std::int64_t stride_b, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size);
    void (*row_major_dgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
        cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b, double beta,
        cl::sycl::buffer<double, 1> &c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size);
    void (*row_major_cgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size);
    void (*row_major_zgemm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
        std::int64_t stride_c, std::int64_t batch_size);
    void (*row_major_strsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::int64_t batch_size);
    void (*row_major_dtrsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
        std::int64_t batch_size);
    void (*row_major_ctrsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
        std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    void (*row_major_ztrsm_batch_strided_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
        std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    void (*row_major_sgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                  std::int64_t n, std::int64_t k, float alpha,
                                  cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                  cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*row_major_dgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                  std::int64_t n, std::int64_t k, double alpha,
                                  cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                  cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*row_major_cgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                  std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                  cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                  std::complex<float> beta,
                                  cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*row_major_zgemmt_sycl)(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                                  oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                                  std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                  cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                  std::complex<double> beta,
                                  cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*row_major_gemm_s8u8s32_bias_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
        cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
        cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
        cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);

    // USM APIs

    cl::sycl::event (*row_major_scasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dzasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dasum_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_saxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x, std::int64_t incx,
        float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_daxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x, std::int64_t incx,
        double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_caxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zaxpy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*row_major_saxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x, std::int64_t *incx,
        float **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*row_major_daxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
        std::int64_t *incx, double **y, std::int64_t *incy, std::int64_t group_count,
        std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*row_major_caxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
        const std::complex<float> **x, std::int64_t *incx, std::complex<float> **y,
        std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*row_major_zaxpy_batch_group_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
        const std::complex<double> **x, std::int64_t *incx, std::complex<double> **y,
        std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

    cl::sycl::event (*row_major_scopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dcopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ccopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zcopy_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sdot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y,
        std::int64_t incy, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ddot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, const double *y,
        std::int64_t incy, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsdot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y,
        std::int64_t incy, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cdotc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zdotc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cdotu_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zdotu_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_isamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_idamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_icamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_izamin_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_isamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_idamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_icamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_izamax_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_scnrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
        float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dznrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
        double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_snrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dnrm2_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_srot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
        std::complex<float> *y, std::int64_t incy, float c, float s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_drot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
        std::complex<double> *y, std::int64_t incy, double c, double s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_csrot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
        std::int64_t incy, float c, float s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zdrot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
        std::int64_t incy, double c, double s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_srotg_usm_sycl)(
        cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_drotg_usm_sycl)(
        cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_crotg_usm_sycl)(
        cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b, float *c,
        std::complex<float> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zrotg_usm_sycl)(
        cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b, double *c,
        std::complex<double> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_srotm_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
        std::int64_t incy, float *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_drotm_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
        std::int64_t incy, double *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_srotmg_usm_sycl)(
        cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1, float *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_drotmg_usm_sycl)(
        cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1, double *param,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float alpha, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double alpha, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, std::complex<float> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_csscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha, std::complex<double> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zdscal_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sdsdot_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx,
        const float *y, std::int64_t incy, float *result,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zswap_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, float alpha, const float *a, std::int64_t lda,
        const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, double alpha, const double *a, std::int64_t lda,
        const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::int64_t kl, std::int64_t ku, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
        std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
        float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        double alpha, const double *a, std::int64_t lda, const double *x, std::int64_t incx,
        double beta, double *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sger_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha, const float *x,
        std::int64_t incx, const float *y, std::int64_t incy, float *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dger_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha, const double *x,
        std::int64_t incx, const double *y, std::int64_t incy, double *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgerc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
        const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
        std::int64_t incy, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgerc_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
        const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
        std::int64_t incy, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgeru_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
        const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
        std::int64_t incy, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgeru_usm_sycl)(
        cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
        const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
        std::int64_t incy, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_chbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zhbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_chemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
        std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zhemv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
        std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cher_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zher_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cher2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zher2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_chpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, const std::complex<float> *x,
        std::int64_t incx, std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zhpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, const std::complex<double> *x,
        std::int64_t incx, std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_chpr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zhpr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_chpr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
        const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zhpr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
        const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ssbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        float alpha, const float *a, std::int64_t lda, const float *x, std::int64_t incx,
        float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
        double alpha, const double *a, std::int64_t lda, const double *x, std::int64_t incx,
        double beta, double *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sspmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *a, const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dspmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *a, const double *x, std::int64_t incx, double beta, double *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sspr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, float *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dspr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, double *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sspr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dspr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ssymv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
        std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsymv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *a, std::int64_t lda, const double *x, std::int64_t incx, double beta,
        double *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ssyr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, float *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsyr_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, double *a, std::int64_t lda,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ssyr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, float alpha,
        const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
        std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsyr2_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n, double alpha,
        const double *x, std::int64_t incx, const double *y, std::int64_t incy, double *a,
        std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_stbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
        std::int64_t lda, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
        std::int64_t lda, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a,
        std::int64_t lda, std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztbmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_stbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
        std::int64_t lda, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
        std::int64_t lda, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<float> *a,
        std::int64_t lda, std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztbsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_stpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztpmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_stpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, double *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztpsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_strmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtrmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctrmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztrmv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_strsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtrsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const double *a, std::int64_t lda, double *x,
        std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctrsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztrsv_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> *x, std::int64_t incx,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a,
        std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a,
        std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
        std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
        std::int64_t ldb, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
        std::int64_t ldb, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_chemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zhemm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cherk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, float alpha, const std::complex<float> *a, std::int64_t lda,
        float beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zherk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, double alpha, const std::complex<double> *a,
        std::int64_t lda, double beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cher2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, float beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zher2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, double beta,
        std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ssymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
        const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda,
        const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_csymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zsymm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ssyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda, float beta,
        float *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
        double beta, double *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_csyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zsyrk_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ssyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
        const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dsyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
        const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_csyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
        std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
        std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zsyr2k_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
        std::int64_t n, std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
        std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_strmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        float alpha, const float *a, std::int64_t lda, float *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtrmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        double alpha, const double *a, std::int64_t lda, double *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctrmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztrmm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        std::complex<double> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_strsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        float alpha, const float *a, std::int64_t lda, float *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dtrsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        double alpha, const double *a, std::int64_t lda, double *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ctrsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
        std::complex<float> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_ztrsm_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
        oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
        std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
        std::complex<double> *b, std::int64_t ldb,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha, const float **a,
        std::int64_t *lda, const float **b, std::int64_t *ldb, float *beta, float **c,
        std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha, const double **a,
        std::int64_t *lda, const double **b, std::int64_t *ldb, double *beta, double **c,
        std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
        const std::complex<float> **a, std::int64_t *lda, const std::complex<float> **b,
        std::int64_t *ldb, std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
        std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgemm_batch_group_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose *transa, oneapi::mkl::transpose *transb,
        std::int64_t *m, std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
        const std::complex<double> **a, std::int64_t *lda, const std::complex<double> **b,
        std::int64_t *ldb, std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
        std::int64_t group_count, std::int64_t *group_size,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a,
        std::int64_t lda, std::int64_t stride_a, const float *b, std::int64_t ldb,
        std::int64_t stride_b, float beta, float *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a,
        std::int64_t lda, std::int64_t stride_a, const double *b, std::int64_t ldb,
        std::int64_t stride_b, double beta, double *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        const std::complex<float> *a, std::int64_t lda, std::int64_t stride_a,
        const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<float> beta, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgemm_batch_strided_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
        std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
        const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
        std::complex<double> beta, std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
        std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_sgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha, const float *a,
        std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_dgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
        const double *a, std::int64_t lda, const double *b, std::int64_t ldb, double beta,
        double *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_cgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
        const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
        std::int64_t ldb, std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);
    cl::sycl::event (*row_major_zgemmt_usm_sycl)(
        cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
        oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
        const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
        std::int64_t ldb, std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
        const cl::sycl::vector_class<cl::sycl::event> &dependencies);

} blas_function_table_t;

#endif //_BLAS_FUNCTION_TABLE_HPP_
