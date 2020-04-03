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
#include "onemkl/types.hpp"

typedef struct {
    int version;
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
    void (*sgbmv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dgbmv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*cgbmv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zgbmv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy);
    void (*sgemv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dgemv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*cgemv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                       std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zgemv_sycl)(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
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
    void (*chbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zhbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy);
    void (*chemv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, std::complex<float> beta,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
    void (*zhemv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, std::complex<double> beta,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);
    void (*cher_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
    void (*zher_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);
    void (*cher2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                       std::int64_t lda);
    void (*zher2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                       std::int64_t lda);
    void (*chpmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy);
    void (*zhpmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy);
    void (*chpr_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*zhpr_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*chpr2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a);
    void (*zhpr2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                       std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a);
    void (*ssbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dsbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*sspmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                       std::int64_t incy);
    void (*dspmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                       std::int64_t incy);
    void (*sspr_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<float, 1> &a);
    void (*dspr_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<double, 1> &a);
    void (*sspr2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a);
    void (*dspr2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a);
    void (*ssymv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy);
    void (*dsymv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy);
    void (*ssyr_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
                      cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*dsyr_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                      double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                      cl::sycl::buffer<double, 1> &a, std::int64_t lda);
    void (*ssyr2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda);
    void (*dsyr2_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
                       double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda);
    void (*stbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztbmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*stbsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtbsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctbsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztbsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*stpmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtpmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctpmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztpmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*stpsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtpsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctpsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztpsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*strmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtrmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctrmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztrmv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*strsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx);
    void (*dtrsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx);
    void (*ctrsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
    void (*ztrsv_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       onemkl::diag unit_diag, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
    void (*sgemm_sycl)(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                       cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dgemm_sycl)(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                       cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*cgemm_sycl)(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zgemm_sycl)(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*hgemm_sycl)(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                       std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                       cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
                       std::int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c, std::int64_t ldc);
    void (*chemm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zhemm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*cherk_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       std::int64_t n, std::int64_t k, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
                       cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zherk_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       std::int64_t n, std::int64_t k, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
                       cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*cher2k_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zher2k_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*ssymm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                       float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dsymm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                       double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*csymm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zsymm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       std::int64_t m, std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*ssyrk_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                       std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c,
                       std::int64_t ldc);
    void (*dsyrk_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c,
                       std::int64_t ldc);
    void (*csyrk_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       std::int64_t n, std::int64_t k, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                       std::int64_t ldc);
    void (*zsyrk_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                       std::int64_t n, std::int64_t k, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                       std::int64_t ldc);
    void (*ssyr2k_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dsyr2k_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*csyr2k_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc);
    void (*zsyr2k_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc);
    void (*strmm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*dtrmm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*ctrmm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*ztrmm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*strsm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<float, 1> &b, std::int64_t ldb);
    void (*dtrsm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                       std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb);
    void (*ctrsm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<float> alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
    void (*ztrsm_sycl)(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                       onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, std::complex<double> alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                       cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);
    void (*sgemm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
        cl::sycl::buffer<onemkl::transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
        cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
        cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
        cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
        cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<float, 1> &beta,
        cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
        std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*dgemm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
        cl::sycl::buffer<onemkl::transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
        cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
        cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
        cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
        cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
        cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
        std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*cgemm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
        cl::sycl::buffer<onemkl::transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
        cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
        cl::sycl::buffer<std::complex<float>, 1> &alpha,
        cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
        cl::sycl::buffer<std::complex<float>, 1> &beta, cl::sycl::buffer<std::complex<float>, 1> &c,
        cl::sycl::buffer<std::int64_t, 1> &ldc, std::int64_t group_count,
        cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*zgemm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
        cl::sycl::buffer<onemkl::transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
        cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
        cl::sycl::buffer<std::complex<double>, 1> &alpha,
        cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
        cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
        cl::sycl::buffer<std::complex<double>, 1> &beta,
        cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
        std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*sgemm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                                     onemkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, float beta,
                                     cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*dgemm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                                     onemkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, double beta,
                                     cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*cgemm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                                     onemkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<float> beta,
                                     cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*zgemm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                                     onemkl::transpose transb, std::int64_t m, std::int64_t n,
                                     std::int64_t k, std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<double> beta,
                                     cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size);
    void (*strsm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
        cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
        cl::sycl::buffer<onemkl::transpose, 1> &trans, cl::sycl::buffer<onemkl::diag, 1> &unit_diag,
        cl::sycl::buffer<std::int64_t, 1> &m, cl::sycl::buffer<std::int64_t, 1> &n,
        cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
        cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
        cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
        cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*dtrsm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
        cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
        cl::sycl::buffer<onemkl::transpose, 1> &trans, cl::sycl::buffer<onemkl::diag, 1> &unit_diag,
        cl::sycl::buffer<std::int64_t, 1> &m, cl::sycl::buffer<std::int64_t, 1> &n,
        cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
        cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
        cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
        cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*ctrsm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
        cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
        cl::sycl::buffer<onemkl::transpose, 1> &trans, cl::sycl::buffer<onemkl::diag, 1> &unit_diag,
        cl::sycl::buffer<std::int64_t, 1> &m, cl::sycl::buffer<std::int64_t, 1> &n,
        cl::sycl::buffer<std::complex<float>, 1> &alpha,
        cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
        std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*ztrsm_batch_group_sycl)(
        cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
        cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
        cl::sycl::buffer<onemkl::transpose, 1> &trans, cl::sycl::buffer<onemkl::diag, 1> &unit_diag,
        cl::sycl::buffer<std::int64_t, 1> &m, cl::sycl::buffer<std::int64_t, 1> &n,
        cl::sycl::buffer<std::complex<double>, 1> &alpha,
        cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
        cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
        std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
    void (*strsm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::side left_right,
                                     onemkl::uplo upper_lower, onemkl::transpose trans,
                                     onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size);
    void (*dtrsm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::side left_right,
                                     onemkl::uplo upper_lower, onemkl::transpose trans,
                                     onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size);
    void (*ctrsm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::side left_right,
                                     onemkl::uplo upper_lower, onemkl::transpose trans,
                                     onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size);
    void (*ztrsm_batch_strided_sycl)(cl::sycl::queue &queue, onemkl::side left_right,
                                     onemkl::uplo upper_lower, onemkl::transpose trans,
                                     onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                                     std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size);
    void (*sgemmt_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
                        onemkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                        cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dgemmt_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
                        onemkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*cgemmt_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
                        onemkl::transpose transb, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                        std::int64_t ldb, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zgemmt_sycl)(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
                        onemkl::transpose transb, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                        std::int64_t ldb, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*gemm_f16f16f32_ext_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                                    onemkl::transpose transb, std::int64_t m, std::int64_t n,
                                    std::int64_t k, float alpha, cl::sycl::buffer<half, 1> &a,
                                    std::int64_t lda, cl::sycl::buffer<half, 1> &b,
                                    std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
                                    std::int64_t ldc);
    void (*gemm_s8u8s32_ext_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                                  onemkl::transpose transb, onemkl::offset offsetc, std::int64_t m,
                                  std::int64_t n, std::int64_t k, float alpha,
                                  cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
                                  cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo,
                                  float beta, cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
                                  cl::sycl::buffer<int32_t, 1> &co);
    void (*sgemm_ext_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                           onemkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                           cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
    void (*dgemm_ext_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                           onemkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                           cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
    void (*cgemm_ext_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                           onemkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                           std::int64_t ldb, std::complex<float> beta,
                           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
    void (*zgemm_ext_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                           onemkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                           std::int64_t ldb, std::complex<double> beta,
                           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
    void (*hgemm_ext_sycl)(cl::sycl::queue &queue, onemkl::transpose transa,
                           onemkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           half alpha, cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                           cl::sycl::buffer<half, 1> &c, std::int64_t ldc);
} function_table_t;

#endif //_BLAS_FUNCTION_TABLE_HPP_
