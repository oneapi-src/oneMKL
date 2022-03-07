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

// Buffer APIs

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &result) {
    ::oneapi::mkl::gpu::scasum(queue, n, x, incx, result);
}

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<double, 1> &result) {
    ::oneapi::mkl::gpu::dzasum(queue, n, x, incx, result);
}

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &result) {
    ::oneapi::mkl::gpu::sasum(queue, n, x, incx, result);
}

void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &result) {
    ::oneapi::mkl::gpu::dasum(queue, n, x, incx, result);
}

void axpy(sycl::queue &queue, std::int64_t n, float alpha, sycl::buffer<float, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::saxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue &queue, std::int64_t n, double alpha, sycl::buffer<double, 1> &x,
          std::int64_t incx, sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::daxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::caxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::zaxpy(queue, n, alpha, x, incx, y, incy);
}

void axpby(sycl::queue &queue, std::int64_t n, float alpha, sycl::buffer<float, 1> &x,
           std::int64_t incx, float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::saxpby_sycl(&queue, n, alpha, &x, incx, beta, &y, incy);
}

void axpby(sycl::queue &queue, std::int64_t n, double alpha, sycl::buffer<double, 1> &x,
           std::int64_t incx, double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::daxpby_sycl(&queue, n, alpha, &x, incx, beta, &y, incy);
}

void axpby(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::caxpby_sycl(&queue, n, alpha, &x, incx, beta, &y, incy);
}

void axpby(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
           std::int64_t incy) {
    ::oneapi::mkl::gpu::zaxpby_sycl(&queue, n, alpha, &x, incx, beta, &y, incy);
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::scopy(queue, n, x, incx, y, incy);
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::dcopy(queue, n, x, incx, y, incy);
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::ccopy(queue, n, x, incx, y, incy);
}

void copy(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::zcopy(queue, n, x, incx, y, incy);
}

void dot(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
         sycl::buffer<float, 1> &y, std::int64_t incy, sycl::buffer<float, 1> &result) {
    ::oneapi::mkl::gpu::sdot(queue, n, x, incx, y, incy, result);
}

void dot(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
         sycl::buffer<double, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &result) {
    ::oneapi::mkl::gpu::ddot(queue, n, x, incx, y, incy, result);
}

void sdsdot(sycl::queue &queue, std::int64_t n, float sb, sycl::buffer<float, 1> &x,
            std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
            sycl::buffer<float, 1> &result) {
    ::oneapi::mkl::gpu::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

void dot(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
         sycl::buffer<float, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &result) {
    ::oneapi::mkl::gpu::dsdot(queue, n, x, incx, y, incy, result);
}

void dotc(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &result) {
    ::oneapi::mkl::gpu::cdotc(queue, n, x, incx, y, incy, result);
}

void dotc(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &result) {
    ::oneapi::mkl::gpu::zdotc(queue, n, x, incx, y, incy, result);
}

void dotu(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1> &result) {
    ::oneapi::mkl::gpu::cdotu(queue, n, x, incx, y, incy, result);
}

void dotu(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1> &result) {
    ::oneapi::mkl::gpu::zdotu(queue, n, x, incx, y, incy, result);
}

void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<float, 1> &result) {
    ::oneapi::mkl::gpu::scnrm2(queue, n, x, incx, result);
}

void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<double, 1> &result) {
    ::oneapi::mkl::gpu::dznrm2(queue, n, x, incx, result);
}

void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &result) {
    ::oneapi::mkl::gpu::snrm2(queue, n, x, incx, result);
}

void nrm2(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &result) {
    ::oneapi::mkl::gpu::dnrm2(queue, n, x, incx, result);
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
         std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
         float s) {
    ::oneapi::mkl::gpu::csrot(queue, n, x, incx, y, incy, c, s);
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
         std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
         double c, double s) {
    ::oneapi::mkl::gpu::zdrot(queue, n, x, incx, y, incy, c, s);
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
         sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    ::oneapi::mkl::gpu::srot(queue, n, x, incx, y, incy, c, s);
}

void rot(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
         sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    ::oneapi::mkl::gpu::drot(queue, n, x, incx, y, incy, c, s);
}

void rotg(sycl::queue &queue, sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &b,
          sycl::buffer<float, 1> &c, sycl::buffer<float, 1> &s) {
    ::oneapi::mkl::gpu::srotg(queue, a, b, c, s);
}

void rotg(sycl::queue &queue, sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &b,
          sycl::buffer<double, 1> &c, sycl::buffer<double, 1> &s) {
    ::oneapi::mkl::gpu::drotg(queue, a, b, c, s);
}

void rotg(sycl::queue &queue, sycl::buffer<std::complex<float>, 1> &a,
          sycl::buffer<std::complex<float>, 1> &b, sycl::buffer<float, 1> &c,
          sycl::buffer<std::complex<float>, 1> &s) {
    ::oneapi::mkl::gpu::crotg(queue, a, b, c, s);
}

void rotg(sycl::queue &queue, sycl::buffer<std::complex<double>, 1> &a,
          sycl::buffer<std::complex<double>, 1> &b, sycl::buffer<double, 1> &c,
          sycl::buffer<std::complex<double>, 1> &s) {
    ::oneapi::mkl::gpu::zrotg(queue, a, b, c, s);
}

void rotm(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &y, std::int64_t incy, sycl::buffer<float, 1> &param) {
    ::oneapi::mkl::gpu::srotm(queue, n, x, incx, y, incy, param);
}

void rotm(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy, sycl::buffer<double, 1> &param) {
    ::oneapi::mkl::gpu::drotm(queue, n, x, incx, y, incy, param);
}

void rotmg(sycl::queue &queue, sycl::buffer<float, 1> &d1, sycl::buffer<float, 1> &d2,
           sycl::buffer<float, 1> &x1, float y1, sycl::buffer<float, 1> &param) {
    ::oneapi::mkl::gpu::srotmg(queue, d1, d2, x1, y1, param);
}

void rotmg(sycl::queue &queue, sycl::buffer<double, 1> &d1, sycl::buffer<double, 1> &d2,
           sycl::buffer<double, 1> &x1, double y1, sycl::buffer<double, 1> &param) {
    ::oneapi::mkl::gpu::drotmg(queue, d1, d2, x1, y1, param);
}

void scal(sycl::queue &queue, std::int64_t n, float alpha, sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::sscal(queue, n, alpha, x, incx);
}

void scal(sycl::queue &queue, std::int64_t n, double alpha, sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::dscal(queue, n, alpha, x, incx);
}

void scal(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::cscal(queue, n, alpha, x, incx);
}

void scal(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::zscal(queue, n, alpha, x, incx);
}

void scal(sycl::queue &queue, std::int64_t n, float alpha,
          sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::csscal(queue, n, alpha, x, incx);
}

void scal(sycl::queue &queue, std::int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::zdscal(queue, n, alpha, x, incx);
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
          sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::sswap(queue, n, x, incx, y, incy);
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
          sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::dswap(queue, n, x, incx, y, incy);
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::cswap(queue, n, x, incx, y, incy);
}

void swap(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::zswap(queue, n, x, incx, y, incy);
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::isamax(queue, n, x, incx, result);
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::idamax(queue, n, x, incx, result);
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::icamax(queue, n, x, incx, result);
}

void iamax(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::izamax(queue, n, x, incx, result);
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::isamin(queue, n, x, incx, result);
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::idamin(queue, n, x, incx, result);
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::icamin(queue, n, x, incx, result);
}

void iamin(sycl::queue &queue, std::int64_t n, sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1> &result) {
    ::oneapi::mkl::gpu::izamin(queue, n, x, incx, result);
}

// USM APIs

sycl::event asum(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scasum_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event asum(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dzasum_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event asum(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sasum_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event asum(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dasum_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpy(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpby(sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                      std::int64_t incx, float beta, float *y, std::int64_t incy,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpby_sycl(&queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event axpby(sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                      std::int64_t incx, double beta, double *y, std::int64_t incy,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpby_sycl(&queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event axpby(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                      const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                      std::complex<float> *y, std::int64_t incy,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpby_sycl(&queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event axpby(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                      const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                      std::complex<double> *y, std::int64_t incy,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpby_sycl(&queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ccopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event copy(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event dot(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, float *result,
                    const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dot(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                    const double *y, std::int64_t incy, double *result,
                    const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ddot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event sdsdot(sycl::queue &queue, std::int64_t n, float sb, const float *x,
                       std::int64_t incx, const float *y, std::int64_t incy, float *result,
                       const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sdsdot_sycl(&queue, n, sb, x, incx, y, incy, result, dependencies);
}

sycl::event dot(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, double *result,
                    const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotc(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotc(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotu(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotu(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dznrm2_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::snrm2_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event nrm2(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event rot(sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                    std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c, float s,
                    const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rot(sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                    std::int64_t incx, std::complex<double> *y, std::int64_t incy, double c,
                    double s, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rot(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                    std::int64_t incy, float c, float s,
                    const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rot(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                    std::int64_t incy, double c, double s,
                    const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rotg(sycl::queue &queue, float *a, float *b, float *c, float *s,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotg_sycl(&queue, a, b, c, s, dependencies);
}

sycl::event rotg(sycl::queue &queue, double *a, double *b, double *c, double *s,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotg_sycl(&queue, a, b, c, s, dependencies);
}

sycl::event rotg(sycl::queue &queue, std::complex<float> *a, std::complex<float> *b,
                     float *c, std::complex<float> *s,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::crotg_sycl(&queue, a, b, c, s, dependencies);
}

sycl::event rotg(sycl::queue &queue, std::complex<double> *a, std::complex<double> *b,
                     double *c, std::complex<double> *s,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zrotg_sycl(&queue, a, b, c, s, dependencies);
}

sycl::event rotm(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, float *param,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

sycl::event rotm(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy, double *param,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

sycl::event rotmg(sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                      float *param, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

sycl::event rotmg(sycl::queue &queue, double *d1, double *d2, double *x1, double y1,
                      double *param, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

sycl::event scal(sycl::queue &queue, std::int64_t n, float alpha, float *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue &queue, std::int64_t n, double alpha, double *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
                     std::int64_t incx, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

sycl::event swap(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event swap(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event swap(sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event swap(sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::isamax_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::idamax_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::icamax_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event iamax(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::izamax_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::isamin_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::idamin_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::icamin_sycl(&queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::izamin_sycl(&queue, n, x, incx, result, dependencies);
}
