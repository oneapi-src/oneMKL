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

#include <CL/sycl.hpp>

#include "cpu_common.hpp"

namespace onemkl {
namespace mklcpu {

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        auto accessor_a   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x   = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y   = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sgbmv>(cgh, [=]() {
            ::sgbmv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const MKL_INT *)&kl, (const MKL_INT *)&ku, (const float *)&alpha,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, (const float *)&beta, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        auto accessor_a   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x   = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y   = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dgbmv>(cgh, [=]() {
            ::dgbmv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const MKL_INT *)&kl, (const MKL_INT *)&ku, (const double *)&alpha,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, (const double *)&beta, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgbmv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_  = { beta_real, beta_imag };
            ::cgbmv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const MKL_INT *)&kl, (const MKL_INT *)&ku, (const MKL_Complex8 *)&alpha_,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, (const MKL_Complex8 *)&beta_, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgbmv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_  = { beta_real, beta_imag };
            ::zgbmv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const MKL_INT *)&kl, (const MKL_INT *)&ku, (const MKL_Complex16 *)&alpha_,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, (const MKL_Complex16 *)&beta_, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        auto accessor_a   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x   = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y   = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sgemv>(cgh, [=]() {
            ::sgemv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const float *)&alpha, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const float *)&beta,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        auto accessor_a   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x   = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y   = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dgemv>(cgh, [=]() {
            ::dgemv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const double *)&alpha, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const double *)&beta,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgemv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_  = { beta_real, beta_imag };
            ::cgemv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const MKL_Complex8 *)&alpha_, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const MKL_Complex8 *)&beta_,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char trans_ = *fortran_char(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgemv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_  = { beta_real, beta_imag };
            ::zgemv((const char *)&trans_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                    (const MKL_Complex16 *)&alpha_, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const MKL_Complex16 *)&beta_,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void ger(cl::sycl::queue &queue, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
         int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &a,
         int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sger>(cgh, [=]() {
            ::sger((const MKL_INT *)&m, (const MKL_INT *)&n, (const float *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                   (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void ger(cl::sycl::queue &queue, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
         int64_t incx, cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &a,
         int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dger>(cgh, [=]() {
            ::dger((const MKL_INT *)&m, (const MKL_INT *)&n, (const double *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                   (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgerc>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cgerc((const MKL_INT *)&m, (const MKL_INT *)&n, (const MKL_Complex8 *)&alpha_,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgerc>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::zgerc((const MKL_INT *)&m, (const MKL_INT *)&n, (const MKL_Complex16 *)&alpha_,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgeru>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cgeru((const MKL_INT *)&m, (const MKL_INT *)&n, (const MKL_Complex8 *)&alpha_,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgeru>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::zgeru((const MKL_INT *)&m, (const MKL_INT *)&n, (const MKL_Complex16 *)&alpha_,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chbmv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_  = { beta_real, beta_imag };
            ::chbmv((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_INT *)&k,
                    (const MKL_Complex8 *)&alpha_, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const MKL_Complex8 *)&beta_,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhbmv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_  = { beta_real, beta_imag };
            ::zhbmv((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_INT *)&k,
                    (const MKL_Complex16 *)&alpha_, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const MKL_Complex16 *)&beta_,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chemv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_  = { beta_real, beta_imag };
            ::chemv((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_Complex8 *)&alpha_,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, (const MKL_Complex8 *)&beta_, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy);
        });
    });
}

void hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhemv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_  = { beta_real, beta_imag };
            ::zhemv((const char *)&upper_lower_, (const MKL_INT *)&n,
                    (const MKL_Complex16 *)&alpha_, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const MKL_Complex16 *)&beta_,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cher>(cgh, [=]() {
            ::cher((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_a.get_pointer(),
                   (const MKL_INT *)&lda);
        });
    });
}

void her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zher>(cgh, [=]() {
            ::zher((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_a.get_pointer(),
                   (const MKL_INT *)&lda);
        });
    });
}

void her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cher2>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cher2((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_Complex8 *)&alpha_,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zher2>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::zher2((const char *)&upper_lower_, (const MKL_INT *)&n,
                    (const MKL_Complex16 *)&alpha_, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, accessor_y.get_pointer(), (const MKL_INT *)&incy,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x  = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y  = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chpmv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_  = { beta_real, beta_imag };
            ::chpmv((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_Complex8 *)&alpha_,
                    accessor_ap.get_pointer(), accessor_x.get_pointer(), (const MKL_INT *)&incx,
                    (const MKL_Complex8 *)&beta_, accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x  = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y  = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhpmv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_  = { beta_real, beta_imag };
            ::zhpmv((const char *)&upper_lower_, (const MKL_INT *)&n,
                    (const MKL_Complex16 *)&alpha_, accessor_ap.get_pointer(),
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const MKL_Complex16 *)&beta_,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chpr>(cgh, [=]() {
            ::chpr((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_ap.get_pointer());
        });
    });
}

void hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhpr>(cgh, [=]() {
            ::zhpr((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_ap.get_pointer());
        });
    });
}

void hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x  = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y  = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chpr2>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::chpr2((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_Complex8 *)&alpha_,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_ap.get_pointer());
        });
    });
}

void hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x  = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y  = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhpr2>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::zhpr2((const char *)&upper_lower_, (const MKL_INT *)&n,
                    (const MKL_Complex16 *)&alpha_, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, accessor_y.get_pointer(), (const MKL_INT *)&incy,
                    accessor_ap.get_pointer());
        });
    });
}

void sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssbmv>(cgh, [=]() {
            ::ssbmv((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_INT *)&k,
                    (const float *)&alpha, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const float *)&beta,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsbmv>(cgh, [=]() {
            ::dsbmv((const char *)&upper_lower_, (const MKL_INT *)&n, (const MKL_INT *)&k,
                    (const double *)&alpha, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, (const double *)&beta,
                    accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sspmv>(cgh, [=]() {
            ::sspmv((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                    accessor_ap.get_pointer(), accessor_x.get_pointer(), (const MKL_INT *)&incx,
                    (const float *)&beta, accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dspmv>(cgh, [=]() {
            ::dspmv((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                    accessor_ap.get_pointer(), accessor_x.get_pointer(), (const MKL_INT *)&incx,
                    (const double *)&beta, accessor_y.get_pointer(), (const MKL_INT *)&incy);
        });
    });
}

void spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sspr>(cgh, [=]() {
            ::sspr((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_ap.get_pointer());
        });
    });
}

void spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dspr>(cgh, [=]() {
            ::dspr((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_ap.get_pointer());
        });
    });
}

void spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sspr2>(cgh, [=]() {
            ::sspr2((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_ap.get_pointer());
        });
    });
}

void spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dspr2>(cgh, [=]() {
            ::dspr2((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_ap.get_pointer());
        });
    });
}

void symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssymv>(cgh, [=]() {
            ::ssymv((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, (const float *)&beta, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy);
        });
    });
}

void symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsymv>(cgh, [=]() {
            ::dsymv((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                    accessor_a.get_pointer(), (const MKL_INT *)&lda, accessor_x.get_pointer(),
                    (const MKL_INT *)&incx, (const double *)&beta, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy);
        });
    });
}

void syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssyr>(cgh, [=]() {
            ::ssyr((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_a.get_pointer(),
                   (const MKL_INT *)&lda);
        });
    });
}

void syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a,
         int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsyr>(cgh, [=]() {
            ::dsyr((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                   accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_a.get_pointer(),
                   (const MKL_INT *)&lda);
        });
    });
}

void syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssyr2>(cgh, [=]() {
            ::ssyr2((const char *)&upper_lower_, (const MKL_INT *)&n, (const float *)&alpha,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y         = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsyr2>(cgh, [=]() {
            ::dsyr2((const char *)&upper_lower_, (const MKL_INT *)&n, (const double *)&alpha,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx, accessor_y.get_pointer(),
                    (const MKL_INT *)&incy, accessor_a.get_pointer(), (const MKL_INT *)&lda);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stbmv>(cgh, [=]() {
            ::stbmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtbmv>(cgh, [=]() {
            ::dtbmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctbmv>(cgh, [=]() {
            ::ctbmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztbmv>(cgh, [=]() {
            ::ztbmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stbsv>(cgh, [=]() {
            ::stbsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtbsv>(cgh, [=]() {
            ::dtbsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctbsv>(cgh, [=]() {
            ::ctbsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztbsv>(cgh, [=]() {
            ::ztbsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, (const MKL_INT *)&k, accessor_a.get_pointer(),
                    (const MKL_INT *)&lda, accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stpmv>(cgh, [=]() {
            ::stpmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtpmv>(cgh, [=]() {
            ::dtpmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctpmv>(cgh, [=]() {
            ::ctpmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztpmv>(cgh, [=]() {
            ::ztpmv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stpsv>(cgh, [=]() {
            ::stpsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtpsv>(cgh, [=]() {
            ::dtpsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctpsv>(cgh, [=]() {
            ::ctpsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_ap        = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztpsv>(cgh, [=]() {
            ::ztpsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                    (const MKL_INT *)&incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_strmv>(cgh, [=]() {
            ::strmv((const char *)&upper_lower_, (const char *)&transa_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_b.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtrmv>(cgh, [=]() {
            ::dtrmv((const char *)&upper_lower_, (const char *)&transa_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_b.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctrmv>(cgh, [=]() {
            ::ctrmv((const char *)&upper_lower_, (const char *)&transa_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_b.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char transa_      = *fortran_char(transa);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b         = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztrmv>(cgh, [=]() {
            ::ztrmv((const char *)&upper_lower_, (const char *)&transa_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_b.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_strsv>(cgh, [=]() {
            ::strsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtrsv>(cgh, [=]() {
            ::dtrsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctrsv>(cgh, [=]() {
            ::ctrsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        const char upper_lower_ = *fortran_char(upper_lower);
        const char trans_       = *fortran_char(trans);
        const char unit_diag_   = *fortran_char(unit_diag);
        auto accessor_a         = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x         = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztrsv>(cgh, [=]() {
            ::ztrsv((const char *)&upper_lower_, (const char *)&trans_, (const char *)&unit_diag_,
                    (const MKL_INT *)&n, accessor_a.get_pointer(), (const MKL_INT *)&lda,
                    accessor_x.get_pointer(), (const MKL_INT *)&incx);
        });
    });
}

} // namespace mklcpu
} // namespace onemkl
