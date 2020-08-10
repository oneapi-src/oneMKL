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

#include "oneapi/mkl/blas/detail/blas_loader.hpp"

#include "function_table_initializer.hpp"
#include "blas/function_table.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace detail {

static oneapi::mkl::detail::table_initializer<domain::blas, function_table_t> function_tables;

// Buffer APIs

void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    function_tables[libkey].scasum_sycl(queue, n, x, incx, result);
}

void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    function_tables[libkey].dzasum_sycl(queue, n, x, incx, result);
}

void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    function_tables[libkey].sasum_sycl(queue, n, x, incx, result);
}

void asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    function_tables[libkey].dasum_sycl(queue, n, x, incx, result);
}

void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].saxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].daxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libkey].caxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    function_tables[libkey].zaxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].scopy_sycl(queue, n, x, incx, y, incy);
}

void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].dcopy_sycl(queue, n, x, incx, y, incy);
}

void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libkey].ccopy_sycl(queue, n, x, incx, y, incy);
}

void copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    function_tables[libkey].zcopy_sycl(queue, n, x, incx, y, incy);
}

void dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &result) {
    function_tables[libkey].sdot_sycl(queue, n, x, incx, y, incy, result);
}

void dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    function_tables[libkey].ddot_sycl(queue, n, x, incx, y, incy, result);
}

void dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    function_tables[libkey].dsdot_sycl(queue, n, x, incx, y, incy, result);
}

void dotc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    function_tables[libkey].cdotc_sycl(queue, n, x, incx, y, incy, result);
}

void dotc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    function_tables[libkey].zdotc_sycl(queue, n, x, incx, y, incy, result);
}

void dotu(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    function_tables[libkey].cdotu_sycl(queue, n, x, incx, y, incy, result);
}

void dotu(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    function_tables[libkey].zdotu_sycl(queue, n, x, incx, y, incy, result);
}

void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].isamin_sycl(queue, n, x, incx, result);
}

void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<double, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].idamin_sycl(queue, n, x, incx, result);
}

void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].icamin_sycl(queue, n, x, incx, result);
}

void iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].izamin_sycl(queue, n, x, incx, result);
}

void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].isamax_sycl(queue, n, x, incx, result);
}

void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<double, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].idamax_sycl(queue, n, x, incx, result);
}

void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].icamax_sycl(queue, n, x, incx, result);
}

void iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libkey].izamax_sycl(queue, n, x, incx, result);
}

void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    function_tables[libkey].snrm2_sycl(queue, n, x, incx, result);
}

void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    function_tables[libkey].dnrm2_sycl(queue, n, x, incx, result);
}

void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    function_tables[libkey].scnrm2_sycl(queue, n, x, incx, result);
}

void nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    function_tables[libkey].dznrm2_sycl(queue, n, x, incx, result);
}

void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c, float s) {
    function_tables[libkey].srot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c, double s) {
    function_tables[libkey].drot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, float c, float s) {
    function_tables[libkey].csrot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, double c, double s) {
    function_tables[libkey].zdrot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<float, 1> &s) {
    function_tables[libkey].srotg_sycl(queue, a, b, c, s);
}

void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<double, 1> &s) {
    function_tables[libkey].drotg_sycl(queue, a, b, c, s);
}

void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<std::complex<float>, 1> &s) {
    function_tables[libkey].crotg_sycl(queue, a, b, c, s);
}

void rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    function_tables[libkey].zrotg_sycl(queue, a, b, c, s);
}

void rotm(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &param) {
    function_tables[libkey].srotm_sycl(queue, n, x, incx, y, incy, param);
}

void rotm(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &param) {
    function_tables[libkey].drotm_sycl(queue, n, x, incx, y, incy, param);
}

void rotmg(oneapi::mkl::device libkey, cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
           cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
           cl::sycl::buffer<float, 1> &param) {
    function_tables[libkey].srotmg_sycl(queue, d1, d2, x1, y1, param);
}

void rotmg(oneapi::mkl::device libkey, cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
           cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1, double y1,
           cl::sycl::buffer<double, 1> &param) {
    function_tables[libkey].drotmg_sycl(queue, d1, d2, x1, y1, param);
}

void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libkey].sscal_sycl(queue, n, alpha, x, incx);
}

void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libkey].dscal_sycl(queue, n, alpha, x, incx);
}

void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx) {
    function_tables[libkey].cscal_sycl(queue, n, alpha, x, incx);
}

void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx) {
    function_tables[libkey].csscal_sycl(queue, n, alpha, x, incx);
}

void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libkey].zscal_sycl(queue, n, alpha, x, incx);
}

void scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libkey].zdscal_sycl(queue, n, alpha, x, incx);
}

void sdsdot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float sb,
            cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
            std::int64_t incy, cl::sycl::buffer<float, 1> &result) {
    function_tables[libkey].sdsdot_sycl(queue, n, sb, x, incx, y, incy, result);
}

void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].sswap_sycl(queue, n, x, incx, y, incy);
}

void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].dswap_sycl(queue, n, x, incx, y, incy);
}

void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libkey].cswap_sycl(queue, n, x, incx, y, incy);
}

void swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    function_tables[libkey].zswap_sycl(queue, n, x, incx, y, incy);
}

void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libkey].sgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libkey].dgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libkey].cgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].zgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libkey].sgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libkey].dgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].cgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].zgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ger(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
         float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    function_tables[libkey].sger_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
         double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    function_tables[libkey].dger_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libkey].cgerc_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libkey].zgerc_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libkey].cgeru_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libkey].zgeru_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].chbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void hbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].zhbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void hemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libkey].chemv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void hemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].zhemv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void her(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         float alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libkey].cher_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void her(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         double alpha, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libkey].zher_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void her2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libkey].cher2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void her2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libkey].zher2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void hpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libkey].chpmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void hpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libkey].zhpmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         float alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    function_tables[libkey].chpr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         double alpha, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    function_tables[libkey].zhpr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void hpr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    function_tables[libkey].chpr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void hpr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    function_tables[libkey].zhpr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void sbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libkey].ssbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void sbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libkey].dsbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void spmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libkey].sspmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libkey].dspmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &a) {
    function_tables[libkey].sspr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void spr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &a) {
    function_tables[libkey].dspr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void spr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &a) {
    function_tables[libkey].sspr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void spr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &a) {
    function_tables[libkey].dspr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void symv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libkey].ssymv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void symv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libkey].dsymv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                       incy);
}

void syr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    function_tables[libkey].ssyr_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
         double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    function_tables[libkey].dsyr_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda) {
    function_tables[libkey].ssyr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void syr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda) {
    function_tables[libkey].dsyr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libkey].stbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libkey].dtbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ctbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ztbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libkey].stbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libkey].dtbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ctbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ztbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libkey].stpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libkey].dtpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ctpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ztpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libkey].stpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libkey].dtpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ctpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ztpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libkey].strmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libkey].dtrmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ctrmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ztrmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libkey].strsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libkey].dtrsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ctrsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libkey].ztrsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libkey].sgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                       c, ldc);
}

void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libkey].dgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                       c, ldc);
}

void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].cgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                       c, ldc);
}

void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].zgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                       c, ldc);
}

void gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa, transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    function_tables[libkey].hgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                       c, ldc);
}

void hemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].chemm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                       beta, c, ldc);
}

void hemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].zhemm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                       beta, c, ldc);
}

void herk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    function_tables[libkey].cherk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                       ldc);
}

void herk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].zherk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                       ldc);
}

void her2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].cher2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void her2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].zher2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libkey].ssymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                       beta, c, ldc);
}

void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libkey].dsymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                       beta, c, ldc);
}

void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].csymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                       beta, c, ldc);
}

void symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].zsymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                       beta, c, ldc);
}

void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libkey].ssyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                       ldc);
}

void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libkey].dsyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                       ldc);
}

void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].csyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                       ldc);
}

void syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].zsyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                       ldc);
}

void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
           std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libkey].ssyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libkey].dsyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].csyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    function_tables[libkey].zsyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    function_tables[libkey].strmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    function_tables[libkey].dtrmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    function_tables[libkey].ctrmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    function_tables[libkey].ztrmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    function_tables[libkey].strsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    function_tables[libkey].dtrsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    function_tables[libkey].ctrsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right, uplo upper_lower,
          transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    function_tables[libkey].ztrsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                       alpha, a, lda, b, ldb);
}

void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    function_tables[libkey].sgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size);
}

void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    function_tables[libkey].dgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size);
}

void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    function_tables[libkey].cgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size);
}

void gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    function_tables[libkey].zgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size);
}

void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    function_tables[libkey].strsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                     unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                     ldb, stride_b, batch_size);
}

void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    function_tables[libkey].dtrsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                     unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                     ldb, stride_b, batch_size);
}

void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    function_tables[libkey].ctrsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                     unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                     ldb, stride_b, batch_size);
}

void trsm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    function_tables[libkey].ztrsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                     unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                     ldb, stride_b, batch_size);
}

void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libkey].sgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                        ldb, beta, c, ldc);
}

void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libkey].dgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                        ldb, beta, c, ldc);
}

void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libkey].cgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                        ldb, beta, c, ldc);
}

void gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    function_tables[libkey].zgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                        ldb, beta, c, ldc);
}

void gemm_ext(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libkey].gemm_f16f16f32_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                    b, ldb, beta, c, ldc);
}

void gemm_ext(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
              transpose transb, offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
              float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
              cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    function_tables[libkey].gemm_s8u8s32_ext_sycl(queue, transa, transb, offsetc, m, n, k, alpha, a,
                                                  lda, ao, b, ldb, bo, beta, c, ldc, co);
}

void gemm_ext(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libkey].sgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc);
}

void gemm_ext(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
              cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
              std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libkey].dgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc);
}

void gemm_ext(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
              std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
              std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
              std::int64_t ldc) {
    function_tables[libkey].cgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc);
}

void gemm_ext(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
              std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
              std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
              std::int64_t ldc) {
    function_tables[libkey].zgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc);
}

void gemm_ext(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
              transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    function_tables[libkey].hgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc);
}

// USM APIs

cl::sycl::event asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].scasum_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dzasum_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const float *x, std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sasum_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const double *x, std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dasum_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     float alpha, const float *x, std::int64_t incx, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].saxpy_usm_sycl(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     double alpha, const double *x, std::int64_t incx, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].daxpy_usm_sycl(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].caxpy_usm_sycl(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zaxpy_usm_sycl(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n,
                           float *alpha, const float **x, std::int64_t *incx, float **y,
                           std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].saxpy_batch_group_usm_sycl(
        queue, n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
}

cl::sycl::event axpy_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n,
                           double *alpha, const double **x, std::int64_t *incx, double **y,
                           std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].daxpy_batch_group_usm_sycl(
        queue, n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
}

cl::sycl::event axpy_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **x,
                           std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].caxpy_batch_group_usm_sycl(
        queue, n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
}

cl::sycl::event axpy_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **x,
                           std::int64_t *incx, std::complex<double> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zaxpy_batch_group_usm_sycl(
        queue, n, alpha, x, incx, y, incy, group_count, group_size, dependencies);
}

cl::sycl::event copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const float *x, std::int64_t incx, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].scopy_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const double *x, std::int64_t incx, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dcopy_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ccopy_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zcopy_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                    float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sdot_usm_sycl(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                    const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                    double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ddot_usm_sycl(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                    double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsdot_usm_sycl(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                     std::int64_t incy, std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cdotc_usm_sycl(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zdotc_usm_sycl(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                     std::int64_t incy, std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cdotu_usm_sycl(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zdotu_usm_sycl(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const float *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].isamin_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const double *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].idamin_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const std::complex<float> *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].icamin_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const std::complex<double> *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].izamin_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const float *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].isamax_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const double *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].idamax_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const std::complex<float> *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].icamax_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                      const std::complex<double> *x, std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].izamax_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<float> *x, std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].snrm2_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const std::complex<double> *x, std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dnrm2_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const float *x, std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].scnrm2_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     const double *x, std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dznrm2_usm_sycl(queue, n, x, incx, result, dependencies);
}

cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                    std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                    std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].srot_usm_sycl(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                    std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                    std::int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].drot_usm_sycl(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float *x,
                    std::int64_t incx, float *y, std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].csrot_usm_sycl(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double *x,
                    std::int64_t incx, double *y, std::int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zdrot_usm_sycl(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue, float *a, float *b,
                     float *c, float *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].srotg_usm_sycl(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue, double *a, double *b,
                     double *c, double *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].drotg_usm_sycl(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::complex<float> *a,
                     std::complex<float> *b, float *c, std::complex<float> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].crotg_usm_sycl(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::complex<double> *a,
                     std::complex<double> *b, double *c, std::complex<double> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zrotg_usm_sycl(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotm(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float *x,
                     std::int64_t incx, float *y, std::int64_t incy, float *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].srotm_usm_sycl(queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotm(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double *x,
                     std::int64_t incx, double *y, std::int64_t incy, double *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].drotm_usm_sycl(queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotmg(oneapi::mkl::device libkey, cl::sycl::queue &queue, float *d1, float *d2,
                      float *x1, float y1, float *param,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].srotmg_usm_sycl(queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event rotmg(oneapi::mkl::device libkey, cl::sycl::queue &queue, double *d1, double *d2,
                      double *x1, double y1, double *param,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].drotmg_usm_sycl(queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     float alpha, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sscal_usm_sycl(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     double alpha, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dscal_usm_sycl(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     std::complex<float> alpha, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cscal_usm_sycl(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     std::complex<double> alpha, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].csscal_usm_sycl(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     float alpha, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zscal_usm_sycl(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     double alpha, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zdscal_usm_sycl(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event sdsdot(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float sb,
                       const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                       float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sdsdot_usm_sycl(queue, n, sb, x, incx, y, incy, result,
                                                   dependencies);
}

cl::sycl::event swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sswap_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n, double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dswap_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cswap_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t n,
                     std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zswap_usm_sycl(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                     const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                     float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sgbmv_usm_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x,
                                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dgbmv_usm_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x,
                                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgbmv_usm_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x,
                                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, std::int64_t kl, std::int64_t ku,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgbmv_usm_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x,
                                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
                     const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sgemv_usm_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta,
                                                  y, incy, dependencies);
}

cl::sycl::event gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, double alpha, const double *a,
                     std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dgemv_usm_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta,
                                                  y, incy, dependencies);
}

cl::sycl::event gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgemv_usm_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta,
                                                  y, incy, dependencies);
}

cl::sycl::event gemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose trans,
                     std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgemv_usm_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta,
                                                  y, incy, dependencies);
}

cl::sycl::event ger(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                    std::int64_t n, float alpha, const float *x, std::int64_t incx, const float *y,
                    std::int64_t incy, float *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sger_usm_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                                 dependencies);
}

cl::sycl::event ger(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                    std::int64_t n, double alpha, const double *x, std::int64_t incx,
                    const double *y, std::int64_t incy, double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dger_usm_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                                 dependencies);
}

cl::sycl::event gerc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgerc_usm_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                                  dependencies);
}

cl::sycl::event gerc(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgerc_usm_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                                  dependencies);
}

cl::sycl::event geru(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgeru_usm_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                                  dependencies);
}

cl::sycl::event geru(oneapi::mkl::device libkey, cl::sycl::queue &queue, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgeru_usm_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                                  dependencies);
}

cl::sycl::event hbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].chbmv_usm_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event hbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zhbmv_usm_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event hemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].chemv_usm_sycl(queue, upper_lower, n, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event hemv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zhemv_usm_sycl(queue, upper_lower, n, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event her(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, float alpha, const std::complex<float> *x, std::int64_t incx,
                    std::complex<float> *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cher_usm_sycl(queue, upper_lower, n, alpha, x, incx, a, lda,
                                                 dependencies);
}

cl::sycl::event her(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, double alpha, const std::complex<double> *x, std::int64_t incx,
                    std::complex<double> *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zher_usm_sycl(queue, upper_lower, n, alpha, x, incx, a, lda,
                                                 dependencies);
}

cl::sycl::event her2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cher2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  lda, dependencies);
}

cl::sycl::event her2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zher2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  lda, dependencies);
}

cl::sycl::event hpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].chpmv_usm_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                                  incy, dependencies);
}

cl::sycl::event hpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zhpmv_usm_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                                  incy, dependencies);
}

cl::sycl::event hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, float alpha, const std::complex<float> *x, std::int64_t incx,
                    std::complex<float> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].chpr_usm_sycl(queue, upper_lower, n, alpha, x, incx, a,
                                                 dependencies);
}

cl::sycl::event hpr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, double alpha, const std::complex<double> *x, std::int64_t incx,
                    std::complex<double> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zhpr_usm_sycl(queue, upper_lower, n, alpha, x, incx, a,
                                                 dependencies);
}

cl::sycl::event hpr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].chpr2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  dependencies);
}

cl::sycl::event hpr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zhpr2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  dependencies);
}

cl::sycl::event sbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                     const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ssbmv_usm_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event sbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsbmv_usm_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event spmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, float alpha, const float *a, const float *x, std::int64_t incx,
                     float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sspmv_usm_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                                  incy, dependencies);
}

cl::sycl::event spmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, double alpha, const double *a, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dspmv_usm_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                                  incy, dependencies);
}

cl::sycl::event spr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, float alpha, const float *x, std::int64_t incx, float *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sspr_usm_sycl(queue, upper_lower, n, alpha, x, incx, a,
                                                 dependencies);
}

cl::sycl::event spr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, double alpha, const double *x, std::int64_t incx, double *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dspr_usm_sycl(queue, upper_lower, n, alpha, x, incx, a,
                                                 dependencies);
}

cl::sycl::event spr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, float alpha, const float *x, std::int64_t incx, const float *y,
                     std::int64_t incy, float *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sspr2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  dependencies);
}

cl::sycl::event spr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, double alpha, const double *x, std::int64_t incx,
                     const double *y, std::int64_t incy, double *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dspr2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  dependencies);
}

cl::sycl::event symv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ssymv_usm_sycl(queue, upper_lower, n, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event symv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsymv_usm_sycl(queue, upper_lower, n, alpha, a, lda, x, incx,
                                                  beta, y, incy, dependencies);
}

cl::sycl::event syr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, float alpha, const float *x, std::int64_t incx, float *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ssyr_usm_sycl(queue, upper_lower, n, alpha, x, incx, a, lda,
                                                 dependencies);
}

cl::sycl::event syr(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                    std::int64_t n, double alpha, const double *x, std::int64_t incx, double *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsyr_usm_sycl(queue, upper_lower, n, alpha, x, incx, a, lda,
                                                 dependencies);
}

cl::sycl::event syr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, float alpha, const float *x, std::int64_t incx, const float *y,
                     std::int64_t incy, float *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ssyr2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  lda, dependencies);
}

cl::sycl::event syr2(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     std::int64_t n, double alpha, const double *x, std::int64_t incx,
                     const double *y, std::int64_t incy, double *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsyr2_usm_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                                  lda, dependencies);
}

cl::sycl::event tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].stbmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const double *a, std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtbmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctbmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztbmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].stbsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const double *a, std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtbsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctbsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztbsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, k, a,
                                                  lda, x, incx, dependencies);
}

cl::sycl::event tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const float *a, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].stpmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtpmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctpmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event tpmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztpmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const float *a, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].stpsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtpsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctpsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event tpsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztpsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, x,
                                                  incx, dependencies);
}

cl::sycl::event trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].strmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtrmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctrmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event trmv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztrmv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].strsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtrsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctrsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event trsv(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, diag unit_diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztrsv_usm_sycl(queue, upper_lower, trans, unit_diag, n, a, lda,
                                                  x, incx, dependencies);
}

cl::sycl::event gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                     transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                     float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sgemm_usm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                                  ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                     transpose transb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                     const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                     double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dgemm_usm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                                  ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                     transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgemm_usm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                                  ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                     transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgemm_usm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                                  ldb, beta, c, ldc, dependencies);
}

cl::sycl::event hemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].chemm_usm_sycl(queue, left_right, upper_lower, m, n, alpha, a,
                                                  lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event hemm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zhemm_usm_sycl(queue, left_right, upper_lower, m, n, alpha, a,
                                                  lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event herk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, std::int64_t n, std::int64_t k, float alpha,
                     const std::complex<float> *a, std::int64_t lda, float beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cherk_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                  beta, c, ldc, dependencies);
}

cl::sycl::event herk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, std::int64_t n, std::int64_t k, double alpha,
                     const std::complex<double> *a, std::int64_t lda, double beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zherk_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                  beta, c, ldc, dependencies);
}

cl::sycl::event her2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, float beta, std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cher2k_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                   b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event her2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose trans, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, double beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zher2k_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                   b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, std::int64_t m, std::int64_t n, float alpha, const float *a,
                     std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ssymm_usm_sycl(queue, left_right, upper_lower, m, n, alpha, a,
                                                  lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                     double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsymm_usm_sycl(queue, left_right, upper_lower, m, n, alpha, a,
                                                  lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].csymm_usm_sycl(queue, left_right, upper_lower, m, n, alpha, a,
                                                  lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zsymm_usm_sycl(queue, left_right, upper_lower, m, n, alpha, a,
                                                  lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, std::int64_t n, std::int64_t k, float alpha, const float *a,
                     std::int64_t lda, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ssyrk_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                  beta, c, ldc, dependencies);
}

cl::sycl::event syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsyrk_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                  beta, c, ldc, dependencies);
}

cl::sycl::event syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].csyrk_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                  beta, c, ldc, dependencies);
}

cl::sycl::event syrk(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                     transpose trans, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zsyrk_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                  beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose trans, std::int64_t n, std::int64_t k, float alpha, const float *a,
                      std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ssyr2k_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                   b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose trans, std::int64_t n, std::int64_t k, double alpha,
                      const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                      double beta, double *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dsyr2k_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                   b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose trans, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].csyr2k_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                   b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose trans, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zsyr2k_usm_sycl(queue, upper_lower, trans, n, k, alpha, a, lda,
                                                   b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].strmm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtrmm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctrmm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trmm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztrmm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].strsm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dtrsm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ctrsm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(oneapi::mkl::device libkey, cl::sycl::queue &queue, side left_right,
                     uplo upper_lower, transpose trans, diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].ztrsm_usm_sycl(queue, left_right, upper_lower, trans, unit_diag,
                                                  m, n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           float *alpha, const float **a, std::int64_t *lda, const float **b,
                           std::int64_t *ldb, float *beta, float **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sgemm_batch_group_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, group_count,
        group_size, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           double *alpha, const double **a, std::int64_t *lda, const double **b,
                           std::int64_t *ldb, double *beta, double **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dgemm_batch_group_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, group_count,
        group_size, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgemm_batch_group_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, group_count,
        group_size, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose *transa,
                           transpose *transb, std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgemm_batch_group_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, group_count,
        group_size, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           float alpha, const float *a, std::int64_t lda, std::int64_t stride_a,
                           const float *b, std::int64_t ldb, std::int64_t stride_b, float beta,
                           float *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sgemm_batch_strided_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           double alpha, const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dgemm_batch_strided_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgemm_batch_strided_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(oneapi::mkl::device libkey, cl::sycl::queue &queue, transpose transa,
                           transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgemm_batch_strided_usm_sycl(
        queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
        stride_c, batch_size, dependencies);
}

cl::sycl::event gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      float alpha, const float *a, std::int64_t lda, const float *b,
                      std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].sgemmt_usm_sycl(queue, upper_lower, transa, transb, n, k, alpha,
                                                   a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      double alpha, const double *a, std::int64_t lda, const double *b,
                      std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].dgemmt_usm_sycl(queue, upper_lower, transa, transb, n, k, alpha,
                                                   a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].cgemmt_usm_sycl(queue, upper_lower, transa, transb, n, k, alpha,
                                                   a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(oneapi::mkl::device libkey, cl::sycl::queue &queue, uplo upper_lower,
                      transpose transa, transpose transb, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return function_tables[libkey].zgemmt_usm_sycl(queue, upper_lower, transa, transb, n, k, alpha,
                                                   a, lda, b, ldb, beta, c, ldc, dependencies);
}

} /*namespace detail */
} /* namespace blas */
} /* namespace mkl */
} /* namespace oneapi */
