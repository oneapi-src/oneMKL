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

#include "loader.hpp"

namespace onemkl {
namespace blas {
namespace detail {

void asum(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    function_tables[libname].scasum_sycl(queue, n, x, incx, result);
}

void asum(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    function_tables[libname].dzasum_sycl(queue, n, x, incx, result);
}

void asum(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    function_tables[libname].sasum_sycl(queue, n, x, incx, result);
}

void asum(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    function_tables[libname].dasum_sycl(queue, n, x, incx, result);
}

void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy) {
    function_tables[libname].saxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy) {
    function_tables[libname].daxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].caxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    function_tables[libname].zaxpy_sycl(queue, n, alpha, x, incx, y, incy);
}

void copy(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libname].scopy_sycl(queue, n, x, incx, y, incy);
}

void copy(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libname].dcopy_sycl(queue, n, x, incx, y, incy);
}

void copy(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].ccopy_sycl(queue, n, x, incx, y, incy);
}

void copy(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    function_tables[libname].zcopy_sycl(queue, n, x, incx, y, incy);
}

void dot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
         std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
         cl::sycl::buffer<float, 1> &result) {
    function_tables[libname].sdot_sycl(queue, n, x, incx, y, incy, result);
}

void dot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
         std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
         cl::sycl::buffer<double, 1> &result) {
    function_tables[libname].ddot_sycl(queue, n, x, incx, y, incy, result);
}

void dot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
         std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
         cl::sycl::buffer<double, 1> &result) {
    function_tables[libname].dsdot_sycl(queue, n, x, incx, y, incy, result);
}

void dotc(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    function_tables[libname].cdotc_sycl(queue, n, x, incx, y, incy, result);
}

void dotc(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    function_tables[libname].zdotc_sycl(queue, n, x, incx, y, incy, result);
}

void dotu(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    function_tables[libname].cdotu_sycl(queue, n, x, incx, y, incy, result);
}

void dotu(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    function_tables[libname].zdotu_sycl(queue, n, x, incx, y, incy, result);
}

void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].isamin_sycl(queue, n, x, incx, result);
}

void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].idamin_sycl(queue, n, x, incx, result);
}

void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].icamin_sycl(queue, n, x, incx, result);
}

void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].izamin_sycl(queue, n, x, incx, result);
}

void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].isamax_sycl(queue, n, x, incx, result);
}

void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].idamax_sycl(queue, n, x, incx, result);
}

void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].icamax_sycl(queue, n, x, incx, result);
}

void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    function_tables[libname].izamax_sycl(queue, n, x, incx, result);
}

void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    function_tables[libname].snrm2_sycl(queue, n, x, incx, result);
}

void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    function_tables[libname].dnrm2_sycl(queue, n, x, incx, result);
}

void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    function_tables[libname].scnrm2_sycl(queue, n, x, incx, result);
}

void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    function_tables[libname].dznrm2_sycl(queue, n, x, incx, result);
}

void rot(char *libname, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c, float s) {
    function_tables[libname].srot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rot(char *libname, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c, double s) {
    function_tables[libname].drot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
         std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    function_tables[libname].csrot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
         std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    function_tables[libname].zdrot_sycl(queue, n, x, incx, y, incy, c, s);
}

void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<float, 1> &s) {
    function_tables[libname].srotg_sycl(queue, a, b, c, s);
}

void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<double, 1> &s) {
    function_tables[libname].drotg_sycl(queue, a, b, c, s);
}

void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s) {
    function_tables[libname].crotg_sycl(queue, a, b, c, s);
}

void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    function_tables[libname].zrotg_sycl(queue, a, b, c, s);
}

void rotm(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
          cl::sycl::buffer<float, 1> &param) {
    function_tables[libname].srotm_sycl(queue, n, x, incx, y, incy, param);
}

void rotm(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
          cl::sycl::buffer<double, 1> &param) {
    function_tables[libname].drotm_sycl(queue, n, x, incx, y, incy, param);
}

void rotmg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
           cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
           cl::sycl::buffer<float, 1> &param) {
    function_tables[libname].srotmg_sycl(queue, d1, d2, x1, y1, param);
}

void rotmg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
           cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1, double y1,
           cl::sycl::buffer<double, 1> &param) {
    function_tables[libname].drotmg_sycl(queue, d1, d2, x1, y1, param);
}

void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libname].sscal_sycl(queue, n, alpha, x, incx);
}

void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libname].dscal_sycl(queue, n, alpha, x, incx);
}

void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].cscal_sycl(queue, n, alpha, x, incx);
}

void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].csscal_sycl(queue, n, alpha, x, incx);
}

void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].zscal_sycl(queue, n, alpha, x, incx);
}

void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].zdscal_sycl(queue, n, alpha, x, incx);
}

void sdsdot(char *libname, cl::sycl::queue &queue, std::int64_t n, float sb,
            cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
            std::int64_t incy, cl::sycl::buffer<float, 1> &result) {
    function_tables[libname].sdsdot_sycl(queue, n, sb, x, incx, y, incy, result);
}

void swap(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libname].sswap_sycl(queue, n, x, incx, y, incy);
}

void swap(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libname].dswap_sycl(queue, n, x, incx, y, incy);
}

void swap(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].cswap_sycl(queue, n, x, incx, y, incy);
}

void swap(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    function_tables[libname].zswap_sycl(queue, n, x, incx, y, incy);
}

void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libname].sgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libname].dgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].cgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libname].zgbmv_sycl(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libname].sgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libname].dgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].cgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libname].zgemv_sycl(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ger(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    function_tables[libname].sger_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    function_tables[libname].dger_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libname].cgerc_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libname].zgerc_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libname].cgeru_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libname].zgeru_sycl(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].chbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void hbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libname].zhbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void hemv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].chemv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void hemv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libname].zhemv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void her(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libname].cher_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void her(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libname].zher_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void her2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    function_tables[libname].cher2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void her2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    function_tables[libname].zher2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void hpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    function_tables[libname].chpmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void hpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    function_tables[libname].zhpmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void hpr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    function_tables[libname].chpr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void hpr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    function_tables[libname].zhpr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void hpr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    function_tables[libname].chpr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void hpr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    function_tables[libname].zhpr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void sbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libname].ssbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void sbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libname].dsbmv_sycl(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void spmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libname].sspmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libname].dspmv_sycl(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a) {
    function_tables[libname].sspr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void spr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a) {
    function_tables[libname].dspr_sycl(queue, upper_lower, n, alpha, x, incx, a);
}

void spr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a) {
    function_tables[libname].sspr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void spr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a) {
    function_tables[libname].dspr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void symv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    function_tables[libname].ssymv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void symv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    function_tables[libname].dsymv_sycl(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                        incy);
}

void syr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    function_tables[libname].ssyr_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    function_tables[libname].dsyr_sycl(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    function_tables[libname].ssyr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void syr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    function_tables[libname].dsyr2_sycl(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libname].stbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libname].dtbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].ctbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].ztbmv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libname].stbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libname].dtbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].ctbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].ztbsv_sycl(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                        incx);
}

void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    function_tables[libname].stpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    function_tables[libname].dtpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].ctpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].ztpmv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    function_tables[libname].stpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    function_tables[libname].dtpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].ctpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].ztpsv_sycl(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libname].strmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libname].dtrmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].ctrmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].ztrmv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    function_tables[libname].strsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    function_tables[libname].dtrsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    function_tables[libname].ctrsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    function_tables[libname].ztrsv_sycl(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libname].sgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc);
}

void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libname].dgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc);
}

void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libname].cgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc);
}

void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libname].zgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc);
}

void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    function_tables[libname].hgemm_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc);
}

void hemm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libname].chemm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void hemm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    function_tables[libname].zhemm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void herk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    function_tables[libname].cherk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                        ldc);
}

void herk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    function_tables[libname].zherk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                        ldc);
}

void her2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           float beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libname].cher2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc);
}

void her2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           double beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    function_tables[libname].zher2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc);
}

void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libname].ssymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libname].dsymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libname].csymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    function_tables[libname].zsymm_sycl(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libname].ssyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                        ldc);
}

void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libname].dsyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                        ldc);
}

void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    function_tables[libname].csyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                        ldc);
}

void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    function_tables[libname].zsyrk_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                        ldc);
}

void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
           cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libname].ssyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc);
}

void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
           cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libname].dsyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc);
}

void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
           std::int64_t ldc) {
    function_tables[libname].csyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc);
}

void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    function_tables[libname].zsyr2k_sycl(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc);
}

void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    function_tables[libname].strmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    function_tables[libname].dtrmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    function_tables[libname].ctrmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    function_tables[libname].ztrmm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    function_tables[libname].strsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    function_tables[libname].dtrsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    function_tables[libname].ctrsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    function_tables[libname].ztrsm_sycl(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                        alpha, a, lda, b, ldb);
}

void gemm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<float, 1> &beta,
                cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].sgemm_batch_group_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                    b, ldb, beta, c, ldc, group_count, group_size);
}

void gemm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
                cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].dgemm_batch_group_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                    b, ldb, beta, c, ldc, group_count, group_size);
}

void gemm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                cl::sycl::buffer<std::complex<float>, 1> &beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].cgemm_batch_group_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                    b, ldb, beta, c, ldc, group_count, group_size);
}

void gemm_batch(
    char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].zgemm_batch_group_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                    b, ldb, beta, c, ldc, group_count, group_size);
}

void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    function_tables[libname].sgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                      stride_a, b, ldb, stride_b, beta, c, ldc,
                                                      stride_c, batch_size);
}

void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    function_tables[libname].dgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                      stride_a, b, ldb, stride_b, beta, c, ldc,
                                                      stride_c, batch_size);
}

void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    function_tables[libname].cgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                      stride_a, b, ldb, stride_b, beta, c, ldc,
                                                      stride_c, batch_size);
}

void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    function_tables[libname].zgemm_batch_strided_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                      stride_a, b, ldb, stride_b, beta, c, ldc,
                                                      stride_c, batch_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
                cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].strsm_batch_group_sycl(queue, left_right, upper_lower, trans,
                                                    unit_diag, m, n, alpha, a, lda, b, ldb,
                                                    group_count, group_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
                cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].dtrsm_batch_group_sycl(queue, left_right, upper_lower, trans,
                                                    unit_diag, m, n, alpha, a, lda, b, ldb,
                                                    group_count, group_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].ctrsm_batch_group_sycl(queue, left_right, upper_lower, trans,
                                                    unit_diag, m, n, alpha, a, lda, b, ldb,
                                                    group_count, group_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::complex<double>, 1> &alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<double>, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
                cl::sycl::buffer<std::int64_t, 1> &group_size) {
    function_tables[libname].ztrsm_batch_group_sycl(queue, left_right, upper_lower, trans,
                                                    unit_diag, m, n, alpha, a, lda, b, ldb,
                                                    group_count, group_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    function_tables[libname].strsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                      unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                      ldb, stride_b, batch_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    function_tables[libname].dtrsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                      unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                      ldb, stride_b, batch_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    function_tables[libname].ctrsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                      unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                      ldb, stride_b, batch_size);
}

void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    function_tables[libname].ztrsm_batch_strided_sycl(queue, left_right, upper_lower, trans,
                                                      unit_diag, m, n, alpha, a, lda, stride_a, b,
                                                      ldb, stride_b, batch_size);
}

void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libname].sgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                         ldb, beta, c, ldc);
}

void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libname].dgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                         ldb, beta, c, ldc);
}

void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    function_tables[libname].cgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                         ldb, beta, c, ldc);
}

void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    function_tables[libname].zgemmt_sycl(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                                         ldb, beta, c, ldc);
}

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libname].gemm_f16f16f32_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     b, ldb, beta, c, ldc);
}

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
              cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    function_tables[libname].gemm_s8u8s32_ext_sycl(queue, transa, transb, offsetc, m, n, k, alpha,
                                                   a, lda, ao, b, ldb, bo, beta, c, ldc, co);
}

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    function_tables[libname].sgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                            beta, c, ldc);
}

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
              cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
              std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    function_tables[libname].dgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                            beta, c, ldc);
}

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
              std::int64_t ldc) {
    function_tables[libname].cgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                            beta, c, ldc);
}

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
              std::int64_t ldc) {
    function_tables[libname].zgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                            beta, c, ldc);
}

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    function_tables[libname].hgemm_ext_sycl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                            beta, c, ldc);
}

} /*namespace detail */
} /* namespace blas */
} /* namespace onemkl */
