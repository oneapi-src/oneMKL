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

// Buffer APIs

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sgbmv>(cgh, [=]() {
            ::cblas_sgbmv(CBLASMAJOR, trans_, m, n, kl, ku, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dgbmv>(cgh, [=]() {
            ::cblas_dgbmv(CBLASMAJOR, trans_, m, n, kl, ku, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgbmv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgbmv(CBLASMAJOR, trans_, m, n, kl, ku, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgbmv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgbmv(CBLASMAJOR, trans_, m, n, kl, ku, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sgemv>(cgh, [=]() {
            ::cblas_sgemv(CBLASMAJOR, trans_, m, n, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dgemv>(cgh, [=]() {
            ::cblas_dgemv(CBLASMAJOR, trans_, m, n, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgemv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgemv(CBLASMAJOR, trans_, m, n, (const void *)&alpha_, accessor_a.get_pointer(),
                          lda, accessor_x.get_pointer(), incx, (const void *)&beta_,
                          accessor_y.get_pointer(), incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgemv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgemv(CBLASMAJOR, trans_, m, n, (const void *)&alpha_, accessor_a.get_pointer(),
                          lda, accessor_x.get_pointer(), incx, (const void *)&beta_,
                          accessor_y.get_pointer(), incy);
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
            ::cblas_sger(CBLASMAJOR, m, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
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
            ::cblas_dger(CBLASMAJOR, m, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
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
            ::cblas_cgerc(CBLASMAJOR, m, n, (const void *)&alpha_, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
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
            ::cblas_zgerc(CBLASMAJOR, m, n, (const void *)&alpha_, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
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
            ::cblas_cgeru(CBLASMAJOR, m, n, (const void *)&alpha_, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
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
            ::cblas_zgeru(CBLASMAJOR, m, n, (const void *)&alpha_, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
        });
    });
}

void hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chbmv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chbmv(CBLASMAJOR, upper_lower_, n, k, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhbmv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhbmv(CBLASMAJOR, upper_lower_, n, k, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chemv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chemv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhemv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhemv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cher>(cgh, [=]() {
            ::cblas_cher(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_a.get_pointer(), lda);
        });
    });
}

void her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zher>(cgh, [=]() {
            ::cblas_zher(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_a.get_pointer(), lda);
        });
    });
}

void her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cher2>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_cher2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_x.get_pointer(), incx, accessor_y.get_pointer(), incy,
                          accessor_a.get_pointer(), lda);
        });
    });
}

void her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zher2>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zher2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_x.get_pointer(), incx, accessor_y.get_pointer(), incy,
                          accessor_a.get_pointer(), lda);
        });
    });
}

void hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chpmv>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chpmv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhpmv>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhpmv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx,
                          (const void *)&beta_, accessor_y.get_pointer(), incy);
        });
    });
}

void hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chpr>(cgh, [=]() {
            ::cblas_chpr(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_ap.get_pointer());
        });
    });
}

void hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhpr>(cgh, [=]() {
            ::cblas_zhpr(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_ap.get_pointer());
        });
    });
}

void hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chpr2>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_chpr2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_x.get_pointer(), incx, accessor_y.get_pointer(), incy,
                          accessor_ap.get_pointer());
        });
    });
}

void hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhpr2>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zhpr2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_,
                          accessor_x.get_pointer(), incx, accessor_y.get_pointer(), incy,
                          accessor_ap.get_pointer());
        });
    });
}

void sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssbmv>(cgh, [=]() {
            ::cblas_ssbmv(CBLASMAJOR, upper_lower_, n, k, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsbmv>(cgh, [=]() {
            ::cblas_dsbmv(CBLASMAJOR, upper_lower_, n, k, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sspmv>(cgh, [=]() {
            ::cblas_sspmv(CBLASMAJOR, upper_lower_, n, alpha, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dspmv>(cgh, [=]() {
            ::cblas_dspmv(CBLASMAJOR, upper_lower_, n, alpha, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sspr>(cgh, [=]() {
            ::cblas_sspr(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_ap.get_pointer());
        });
    });
}

void spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dspr>(cgh, [=]() {
            ::cblas_dspr(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_ap.get_pointer());
        });
    });
}

void spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sspr2>(cgh, [=]() {
            ::cblas_sspr2(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_ap.get_pointer());
        });
    });
}

void spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dspr2>(cgh, [=]() {
            ::cblas_dspr2(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_ap.get_pointer());
        });
    });
}

void symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssymv>(cgh, [=]() {
            ::cblas_ssymv(CBLASMAJOR, upper_lower_, n, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsymv>(cgh, [=]() {
            ::cblas_dsymv(CBLASMAJOR, upper_lower_, n, alpha, accessor_a.get_pointer(), lda,
                          accessor_x.get_pointer(), incx, beta, accessor_y.get_pointer(), incy);
        });
    });
}

void syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssyr>(cgh, [=]() {
            ::cblas_ssyr(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_a.get_pointer(), lda);
        });
    });
}

void syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a,
         int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsyr>(cgh, [=]() {
            ::cblas_dsyr(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                         accessor_a.get_pointer(), lda);
        });
    });
}

void syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssyr2>(cgh, [=]() {
            ::cblas_ssyr2(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
        });
    });
}

void syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsyr2>(cgh, [=]() {
            ::cblas_dsyr2(CBLASMAJOR, upper_lower_, n, alpha, accessor_x.get_pointer(), incx,
                          accessor_y.get_pointer(), incy, accessor_a.get_pointer(), lda);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stbmv>(cgh, [=]() {
            ::cblas_stbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtbmv>(cgh, [=]() {
            ::cblas_dtbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctbmv>(cgh, [=]() {
            ::cblas_ctbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztbmv>(cgh, [=]() {
            ::cblas_ztbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stbsv>(cgh, [=]() {
            ::cblas_stbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtbsv>(cgh, [=]() {
            ::cblas_dtbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctbsv>(cgh, [=]() {
            ::cblas_ctbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztbsv>(cgh, [=]() {
            ::cblas_ztbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k,
                          accessor_a.get_pointer(), lda, accessor_x.get_pointer(), incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stpmv>(cgh, [=]() {
            ::cblas_stpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtpmv>(cgh, [=]() {
            ::cblas_dtpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctpmv>(cgh, [=]() {
            ::cblas_ctpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztpmv>(cgh, [=]() {
            ::cblas_ztpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_stpsv>(cgh, [=]() {
            ::cblas_stpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtpsv>(cgh, [=]() {
            ::cblas_dtpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctpsv>(cgh, [=]() {
            ::cblas_ctpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztpsv>(cgh, [=]() {
            ::cblas_ztpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n,
                          accessor_ap.get_pointer(), accessor_x.get_pointer(), incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_strmv>(cgh, [=]() {
            ::cblas_strmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtrmv>(cgh, [=]() {
            ::cblas_dtrmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctrmv>(cgh, [=]() {
            ::cblas_ctrmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztrmv>(cgh, [=]() {
            ::cblas_ztrmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_strsv>(cgh, [=]() {
            ::cblas_strsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, accessor_a.get_pointer(),
                          lda, accessor_x.get_pointer(), incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtrsv>(cgh, [=]() {
            ::cblas_dtrsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, accessor_a.get_pointer(),
                          lda, accessor_x.get_pointer(), incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctrsv>(cgh, [=]() {
            ::cblas_ctrsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, accessor_a.get_pointer(),
                          lda, accessor_x.get_pointer(), incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztrsv>(cgh, [=]() {
            ::cblas_ztrsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, accessor_a.get_pointer(),
                          lda, accessor_x.get_pointer(), incx);
        });
    });
}

// USM APIs

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, float alpha, const float *a, int64_t lda, const float *x,
                     int64_t incx, float beta, float *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_sgbmv_usm>(cgh, [=]() {
            ::cblas_sgbmv(CBLASMAJOR, trans_, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, double alpha, const double *a, int64_t lda, const double *x,
                     int64_t incx, double beta, double *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_dgbmv_usm>(cgh, [=]() {
            ::cblas_dgbmv(CBLASMAJOR, trans_, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *x, int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_cgbmv_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgbmv(CBLASMAJOR, trans_, m, n, kl, ku, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, const std::complex<double> *x, int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zgbmv_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgbmv(CBLASMAJOR, trans_, m, n, kl, ku, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                     const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                     float *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_sgemv_usm>(cgh, [=]() {
            ::cblas_sgemv(CBLASMAJOR, trans_, m, n, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                     const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                     double *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_dgemv_usm>(cgh, [=]() {
            ::cblas_dgemv(CBLASMAJOR, trans_, m, n, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                     const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_cgemv_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgemv(CBLASMAJOR, trans_, m, n, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zgemv_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgemv(CBLASMAJOR, trans_, m, n, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event ger(cl::sycl::queue &queue, int64_t m, int64_t n, float alpha, const float *x,
                    int64_t incx, const float *y, int64_t incy, float *a, int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_sger_usm>(
            cgh, [=]() { ::cblas_sger(CBLASMAJOR, m, n, alpha, x, incx, y, incy, a, lda); });
    });
    return done;
}

cl::sycl::event ger(cl::sycl::queue &queue, int64_t m, int64_t n, double alpha, const double *x,
                    int64_t incx, const double *y, int64_t incy, double *a, int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_dger_usm>(
            cgh, [=]() { ::cblas_dger(CBLASMAJOR, m, n, alpha, x, incx, y, incy, a, lda); });
    });
    return done;
}

cl::sycl::event gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *a, int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_cgerc_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_cgerc(CBLASMAJOR, m, n, (const void *)&alpha_, x, incx, y, incy, a, lda);
        });
    });
    return done;
}

cl::sycl::event gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, int64_t incx, const std::complex<double> *y,
                     int64_t incy, std::complex<double> *a, int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_zgerc_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zgerc(CBLASMAJOR, m, n, (const void *)&alpha_, x, incx, y, incy, a, lda);
        });
    });
    return done;
}

cl::sycl::event geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *a, int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_cgeru_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_cgeru(CBLASMAJOR, m, n, (const void *)&alpha_, x, incx, y, incy, a, lda);
        });
    });
    return done;
}

cl::sycl::event geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, int64_t incx, const std::complex<double> *y,
                     int64_t incy, std::complex<double> *a, int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_zgeru_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zgeru(CBLASMAJOR, m, n, (const void *)&alpha_, x, incx, y, incy, a, lda);
        });
    });
    return done;
}

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                     const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_chbmv_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chbmv(CBLASMAJOR, upper_lower_, n, k, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zhbmv_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhbmv(CBLASMAJOR, upper_lower_, n, k, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, int64_t lda, const std::complex<float> *x,
                     int64_t incx, std::complex<float> beta, std::complex<float> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_chemv_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chemv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zhemv_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhemv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, a, lda, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const std::complex<float> *x, int64_t incx, std::complex<float> *a, int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_cher_usm>(
            cgh, [=]() { ::cblas_cher(CBLASMAJOR, upper_lower_, n, alpha, x, incx, a, lda); });
    });
    return done;
}

cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const std::complex<double> *x, int64_t incx, std::complex<double> *a,
                    int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_zher_usm>(
            cgh, [=]() { ::cblas_zher(CBLASMAJOR, upper_lower_, n, alpha, x, incx, a, lda); });
    });
    return done;
}

cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *a, int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_cher2_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_cher2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, x, incx, y, incy, a,
                          lda);
        });
    });
    return done;
}

cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                     const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                     int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_zher2_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zher2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, x, incx, y, incy, a,
                          lda);
        });
    });
    return done;
}

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *ap, const std::complex<float> *x, int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_chpmv_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chpmv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, ap, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *ap,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zhpmv_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhpmv(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, ap, x, incx,
                          (const void *)&beta_, y, incy);
        });
    });
    return done;
}

cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const std::complex<float> *x, int64_t incx, std::complex<float> *ap,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_chpr_usm>(
            cgh, [=]() { ::cblas_chpr(CBLASMAJOR, upper_lower_, n, alpha, x, incx, ap); });
    });
    return done;
}

cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const std::complex<double> *x, int64_t incx, std::complex<double> *ap,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_zhpr_usm>(
            cgh, [=]() { ::cblas_zhpr(CBLASMAJOR, upper_lower_, n, alpha, x, incx, ap); });
    });
    return done;
}

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *ap,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_chpr2_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_chpr2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, x, incx, y, incy, ap);
        });
    });
    return done;
}

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                     const std::complex<double> *y, int64_t incy, std::complex<double> *ap,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_zhpr2_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zhpr2(CBLASMAJOR, upper_lower_, n, (const void *)&alpha_, x, incx, y, incy, ap);
        });
    });
    return done;
}

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
                     const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                     float *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_ssbmv_usm>(cgh, [=]() {
            ::cblas_ssbmv(CBLASMAJOR, upper_lower_, n, k, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
                     const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                     double *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dsbmv_usm>(cgh, [=]() {
            ::cblas_dsbmv(CBLASMAJOR, upper_lower_, n, k, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *ap, const float *x, int64_t incx, float beta, float *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_sspmv_usm>(cgh, [=]() {
            ::cblas_sspmv(CBLASMAJOR, upper_lower_, n, alpha, ap, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *ap, const double *x, int64_t incx, double beta, double *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dspmv_usm>(cgh, [=]() {
            ::cblas_dspmv(CBLASMAJOR, upper_lower_, n, alpha, ap, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const float *x, int64_t incx, float *ap,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_sspr_usm>(
            cgh, [=]() { ::cblas_sspr(CBLASMAJOR, upper_lower_, n, alpha, x, incx, ap); });
    });
    return done;
}

cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const double *x, int64_t incx, double *ap,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dspr_usm>(
            cgh, [=]() { ::cblas_dspr(CBLASMAJOR, upper_lower_, n, alpha, x, incx, ap); });
    });
    return done;
}

cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *x, int64_t incx, const float *y, int64_t incy, float *ap,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_sspr2_usm>(cgh, [=]() {
            ::cblas_sspr2(CBLASMAJOR, upper_lower_, n, alpha, x, incx, y, incy, ap);
        });
    });
    return done;
}

cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *x, int64_t incx, const double *y, int64_t incy, double *ap,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dspr2_usm>(cgh, [=]() {
            ::cblas_dspr2(CBLASMAJOR, upper_lower_, n, alpha, x, incx, y, incy, ap);
        });
    });
    return done;
}

cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                     float *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_ssymv_usm>(cgh, [=]() {
            ::cblas_ssymv(CBLASMAJOR, upper_lower_, n, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                     double *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dsymv_usm>(cgh, [=]() {
            ::cblas_dsymv(CBLASMAJOR, upper_lower_, n, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const float *x, int64_t incx, float *a, int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_ssyr_usm>(
            cgh, [=]() { ::cblas_ssyr(CBLASMAJOR, upper_lower_, n, alpha, x, incx, a, lda); });
    });
    return done;
}

cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const double *x, int64_t incx, double *a, int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dsyr_usm>(
            cgh, [=]() { ::cblas_dsyr(CBLASMAJOR, upper_lower_, n, alpha, x, incx, a, lda); });
    });
    return done;
}

cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *x, int64_t incx, const float *y, int64_t incy, float *a,
                     int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_ssyr2_usm>(cgh, [=]() {
            ::cblas_ssyr2(CBLASMAJOR, upper_lower_, n, alpha, x, incx, y, incy, a, lda);
        });
    });
    return done;
}

cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *x, int64_t incx, const double *y, int64_t incy, double *a,
                     int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dsyr2_usm>(cgh, [=]() {
            ::cblas_dsyr2(CBLASMAJOR, upper_lower_, n, alpha, x, incx, y, incy, a, lda);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const float *a, int64_t lda, float *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_stbmv_usm>(cgh, [=]() {
            ::cblas_stbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const double *a, int64_t lda, double *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtbmv_usm>(cgh, [=]() {
            ::cblas_dtbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<float> *a, int64_t lda,
                     std::complex<float> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ctbmv_usm>(cgh, [=]() {
            ::cblas_ctbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<double> *a, int64_t lda,
                     std::complex<double> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ztbmv_usm>(cgh, [=]() {
            ::cblas_ztbmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const float *a, int64_t lda, float *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_stbsv_usm>(cgh, [=]() {
            ::cblas_stbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const double *a, int64_t lda, double *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtbsv_usm>(cgh, [=]() {
            ::cblas_dtbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<float> *a, int64_t lda,
                     std::complex<float> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ctbsv_usm>(cgh, [=]() {
            ::cblas_ctbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<double> *a, int64_t lda,
                     std::complex<double> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ztbsv_usm>(cgh, [=]() {
            ::cblas_ztbsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const float *ap, float *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_stpmv_usm>(cgh, [=]() {
            ::cblas_stpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const double *ap, double *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtpmv_usm>(cgh, [=]() {
            ::cblas_dtpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<float> *ap, std::complex<float> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ctpmv_usm>(cgh, [=]() {
            ::cblas_ctpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<double> *ap, std::complex<double> *x,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ztpmv_usm>(cgh, [=]() {
            ::cblas_ztpmv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const float *ap, float *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_stpsv_usm>(cgh, [=]() {
            ::cblas_stpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const double *ap, double *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtpsv_usm>(cgh, [=]() {
            ::cblas_dtpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<float> *ap, std::complex<float> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ctpsv_usm>(cgh, [=]() {
            ::cblas_ctpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<double> *ap, std::complex<double> *x,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ztpsv_usm>(cgh, [=]() {
            ::cblas_ztpsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const float *a, int64_t lda, float *b, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_strmv_usm>(cgh, [=]() {
            ::cblas_strmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n, a, lda, b, incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const double *a, int64_t lda, double *b, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtrmv_usm>(cgh, [=]() {
            ::cblas_dtrmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n, a, lda, b, incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const std::complex<float> *a, int64_t lda, std::complex<float> *b,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ctrmv_usm>(cgh, [=]() {
            ::cblas_ctrmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n, a, lda, b, incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ztrmv_usm>(cgh, [=]() {
            ::cblas_ztrmv(CBLASMAJOR, upper_lower_, transa_, unit_diag_, n, a, lda, b, incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const float *a, int64_t lda, float *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_strsv_usm>(cgh, [=]() {
            ::cblas_strsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const double *a, int64_t lda, double *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtrsv_usm>(cgh, [=]() {
            ::cblas_dtrsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<float> *a, int64_t lda, std::complex<float> *x,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ctrsv_usm>(cgh, [=]() {
            ::cblas_ctrsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, a, lda, x, incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<double> *a, int64_t lda, std::complex<double> *x,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_ztrsv_usm>(cgh, [=]() {
            ::cblas_ztrsv(CBLASMAJOR, upper_lower_, trans_, unit_diag_, n, a, lda, x, incx);
        });
    });
    return done;
}
