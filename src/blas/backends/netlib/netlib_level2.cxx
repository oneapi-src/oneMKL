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

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sgbmv>(cgh, [=]() {
            ::cblas_sgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const float)alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx, (const float)beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dgbmv>(cgh, [=]() {
            ::cblas_dgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const double)alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx, (const double)beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cgbmv>(cgh, [=]() {
            ::cblas_cgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void *)&alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx, (const void *)&beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zgbmv>(cgh, [=]() {
            ::cblas_zgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void *)&alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx, (const void *)&beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sgemv>(cgh, [=]() {
            ::cblas_sgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const float)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const float)beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dgemv>(cgh, [=]() {
            ::cblas_dgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const double)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const double)beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cgemv>(cgh, [=]() {
            ::cblas_cgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const void *)&beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zgemv>(cgh, [=]() {
            ::cblas_zgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const void *)&beta,
                          accessor_y.get_pointer(), (const int)incy);
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
        host_task<class netlib_sger>(cgh, [=]() {
            ::cblas_sger(MAJOR, (const int)m, (const int)n, (const float)alpha,
                         accessor_x.get_pointer(), (const int)incx, accessor_y.get_pointer(),
                         (const int)incy, accessor_a.get_pointer(), (const int)lda);
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
        host_task<class netlib_dger>(cgh, [=]() {
            ::cblas_dger(MAJOR, (const int)m, (const int)n, (const double)alpha,
                         accessor_x.get_pointer(), (const int)incx, accessor_y.get_pointer(),
                         (const int)incy, accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cgerc>(cgh, [=]() {
            ::cblas_cgerc(MAJOR, (const int)m, (const int)n, (const void *)&alpha,
                          accessor_x.get_pointer(), (const int)incx, accessor_y.get_pointer(),
                          (const int)incy, accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zgerc>(cgh, [=]() {
            ::cblas_zgerc(MAJOR, (const int)m, (const int)n, (const void *)&alpha,
                          accessor_x.get_pointer(), (const int)incx, accessor_y.get_pointer(),
                          (const int)incy, accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cgeru>(cgh, [=]() {
            ::cblas_cgeru(MAJOR, (const int)m, (const int)n, (const void *)&alpha,
                          accessor_x.get_pointer(), (const int)incx, accessor_y.get_pointer(),
                          (const int)incy, accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zgeru>(cgh, [=]() {
            ::cblas_zgeru(MAJOR, (const int)m, (const int)n, (const void *)&alpha,
                          accessor_x.get_pointer(), (const int)incx, accessor_y.get_pointer(),
                          (const int)incy, accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_chbmv>(cgh, [=]() {
            ::cblas_chbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const void *)&beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zhbmv>(cgh, [=]() {
            ::cblas_zhbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const void *)&beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_chemv>(cgh, [=]() {
            ::cblas_chemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const void *)&beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zhemv>(cgh, [=]() {
            ::cblas_zhemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const void *)&beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cher>(cgh, [=]() {
            ::cblas_cher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zher>(cgh, [=]() {
            ::cblas_zher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cher2>(cgh, [=]() {
            ::cblas_cher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_a.get_pointer(),
                          (const int)lda);
        });
    });
}

void her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zher2>(cgh, [=]() {
            ::cblas_zher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_a.get_pointer(),
                          (const int)lda);
        });
    });
}

void hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_chpmv>(cgh, [=]() {
            ::cblas_chpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                          (const int)incx, (const void *)&beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zhpmv>(cgh, [=]() {
            ::cblas_zhpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                          (const int)incx, (const void *)&beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_chpr>(cgh, [=]() {
            ::cblas_chpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_ap.get_pointer());
        });
    });
}

void hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zhpr>(cgh, [=]() {
            ::cblas_zhpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_ap.get_pointer());
        });
    });
}

void hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_chpr2>(cgh, [=]() {
            ::cblas_chpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_ap.get_pointer());
        });
    });
}

void hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zhpr2>(cgh, [=]() {
            ::cblas_zhpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_ap.get_pointer());
        });
    });
}

void sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ssbmv>(cgh, [=]() {
            ::cblas_ssbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const float)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const float)beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dsbmv>(cgh, [=]() {
            ::cblas_dsbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const double)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const double)beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sspmv>(cgh, [=]() {
            ::cblas_sspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                          (const int)incx, (const float)beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dspmv>(cgh, [=]() {
            ::cblas_dspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_ap.get_pointer(), accessor_x.get_pointer(),
                          (const int)incx, (const double)beta, accessor_y.get_pointer(),
                          (const int)incy);
        });
    });
}

void spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sspr>(cgh, [=]() {
            ::cblas_sspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_ap.get_pointer());
        });
    });
}

void spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dspr>(cgh, [=]() {
            ::cblas_dspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_ap.get_pointer());
        });
    });
}

void spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sspr2>(cgh, [=]() {
            ::cblas_sspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_ap.get_pointer());
        });
    });
}

void spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &ap) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dspr2>(cgh, [=]() {
            ::cblas_dspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_ap.get_pointer());
        });
    });
}

void symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ssymv>(cgh, [=]() {
            ::cblas_ssymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const float)beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dsymv>(cgh, [=]() {
            ::cblas_dsymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_x.get_pointer(), (const int)incx, (const double)beta,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ssyr>(cgh, [=]() {
            ::cblas_ssyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a,
         int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dsyr>(cgh, [=]() {
            ::cblas_dsyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.get_pointer(), (const int)incx,
                         accessor_a.get_pointer(), (const int)lda);
        });
    });
}

void syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ssyr2>(cgh, [=]() {
            ::cblas_ssyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_a.get_pointer(),
                          (const int)lda);
        });
    });
}

void syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dsyr2>(cgh, [=]() {
            ::cblas_dsyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_a.get_pointer(),
                          (const int)lda);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_stbmv>(cgh, [=]() {
            ::cblas_stbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtbmv>(cgh, [=]() {
            ::cblas_dtbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctbmv>(cgh, [=]() {
            ::cblas_ctbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztbmv>(cgh, [=]() {
            ::cblas_ztbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_stbsv>(cgh, [=]() {
            ::cblas_stbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtbsv>(cgh, [=]() {
            ::cblas_dtbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctbsv>(cgh, [=]() {
            ::cblas_ctbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztbsv>(cgh, [=]() {
            ::cblas_ztbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.get_pointer(), (const int)lda, accessor_x.get_pointer(),
                          (const int)incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_stpmv>(cgh, [=]() {
            ::cblas_stpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtpmv>(cgh, [=]() {
            ::cblas_dtpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctpmv>(cgh, [=]() {
            ::cblas_ctpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztpmv>(cgh, [=]() {
            ::cblas_ztpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &ap, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_stpsv>(cgh, [=]() {
            ::cblas_stpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &ap, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtpsv>(cgh, [=]() {
            ::cblas_dtpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &ap, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctpsv>(cgh, [=]() {
            ::cblas_ctpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &ap,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_ap = ap.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztpsv>(cgh, [=]() {
            ::cblas_ztpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.get_pointer(),
                          accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_strmv>(cgh, [=]() {
            ::cblas_strmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_b.get_pointer(), (const int)incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtrmv>(cgh, [=]() {
            ::cblas_dtrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_b.get_pointer(), (const int)incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctrmv>(cgh, [=]() {
            ::cblas_ctrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_b.get_pointer(), (const int)incx);
        });
    });
}

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztrmv>(cgh, [=]() {
            ::cblas_ztrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_b.get_pointer(), (const int)incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_strsv>(cgh, [=]() {
            ::cblas_strsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtrsv>(cgh, [=]() {
            ::cblas_dtrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctrsv>(cgh, [=]() {
            ::cblas_ctrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_x.get_pointer(), (const int)incx);
        });
    });
}

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztrsv>(cgh, [=]() {
            ::cblas_ztrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.get_pointer(),
                          (const int)lda, accessor_x.get_pointer(), (const int)incx);
        });
    });
}

// USM APIs

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, float alpha, const float *a, int64_t lda, const float *x,
                     int64_t incx, float beta, float *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sgbmv_usm>(cgh, [=]() {
            ::cblas_sgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const float)alpha, a, (const int)lda, x,
                          (const int)incx, (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, double alpha, const double *a, int64_t lda, const double *x,
                     int64_t incx, double beta, double *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dgbmv_usm>(cgh, [=]() {
            ::cblas_dgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const double)alpha, a, (const int)lda, x,
                          (const int)incx, (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *x, int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cgbmv_usm>(cgh, [=]() {
            ::cblas_cgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void *)&alpha, a, (const int)lda, x,
                          (const int)incx, (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, int64_t kl,
                     int64_t ku, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, const std::complex<double> *x, int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zgbmv_usm>(cgh, [=]() {
            ::cblas_zgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void *)&alpha, a, (const int)lda, x,
                          (const int)incx, (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                     const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                     float *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sgemv_usm>(cgh, [=]() {
            ::cblas_sgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const float)alpha, a, (const int)lda, x, (const int)incx,
                          (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                     const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                     double *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dgemv_usm>(cgh, [=]() {
            ::cblas_dgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const double)alpha, a, (const int)lda, x, (const int)incx,
                          (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                     const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cgemv_usm>(cgh, [=]() {
            ::cblas_cgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zgemv_usm>(cgh, [=]() {
            ::cblas_zgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event ger(cl::sycl::queue &queue, int64_t m, int64_t n, float alpha, const float *x,
                    int64_t incx, const float *y, int64_t incy, float *a, int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sger_usm>(cgh, [=]() {
            ::cblas_sger(MAJOR, (const int)m, (const int)n, (const float)alpha, x, (const int)incx,
                         y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event ger(cl::sycl::queue &queue, int64_t m, int64_t n, double alpha, const double *x,
                    int64_t incx, const double *y, int64_t incy, double *a, int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dger_usm>(cgh, [=]() {
            ::cblas_dger(MAJOR, (const int)m, (const int)n, (const double)alpha, x, (const int)incx,
                         y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *a, int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cgerc_usm>(cgh, [=]() {
            ::cblas_cgerc(MAJOR, (const int)m, (const int)n, (const void *)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, int64_t incx, const std::complex<double> *y,
                     int64_t incy, std::complex<double> *a, int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zgerc_usm>(cgh, [=]() {
            ::cblas_zgerc(MAJOR, (const int)m, (const int)n, (const void *)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *a, int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cgeru_usm>(cgh, [=]() {
            ::cblas_cgeru(MAJOR, (const int)m, (const int)n, (const void *)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, int64_t incx, const std::complex<double> *y,
                     int64_t incy, std::complex<double> *a, int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zgeru_usm>(cgh, [=]() {
            ::cblas_zgeru(MAJOR, (const int)m, (const int)n, (const void *)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                     const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_chbmv_usm>(cgh, [=]() {
            ::cblas_chbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void *)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zhbmv_usm>(cgh, [=]() {
            ::cblas_zhbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void *)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, int64_t lda, const std::complex<float> *x,
                     int64_t incx, std::complex<float> beta, std::complex<float> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_chemv_usm>(cgh, [=]() {
            ::cblas_chemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zhemv_usm>(cgh, [=]() {
            ::cblas_zhemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void *)&beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const std::complex<float> *x, int64_t incx, std::complex<float> *a, int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cher_usm>(cgh, [=]() {
            ::cblas_cher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const std::complex<double> *x, int64_t incx, std::complex<double> *a,
                    int64_t lda, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zher_usm>(cgh, [=]() {
            ::cblas_zher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *a, int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cher2_usm>(cgh, [=]() {
            ::cblas_cher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                     const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                     int64_t lda, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zher2_usm>(cgh, [=]() {
            ::cblas_zher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *ap, const std::complex<float> *x, int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_chpmv_usm>(cgh, [=]() {
            ::cblas_chpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, ap, x, (const int)incx, (const void *)&beta, y,
                          (const int)incy);
        });
    });
    return done;
}

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *ap,
                     const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zhpmv_usm>(cgh, [=]() {
            ::cblas_zhpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, ap, x, (const int)incx, (const void *)&beta, y,
                          (const int)incy);
        });
    });
    return done;
}

cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const std::complex<float> *x, int64_t incx, std::complex<float> *ap,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_chpr_usm>(cgh, [=]() {
            ::cblas_chpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const std::complex<double> *x, int64_t incx, std::complex<double> *ap,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zhpr_usm>(cgh, [=]() {
            ::cblas_zhpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, const std::complex<float> *y,
                     int64_t incy, std::complex<float> *ap,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_chpr2_usm>(cgh, [=]() {
            ::cblas_chpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                     const std::complex<double> *y, int64_t incy, std::complex<double> *ap,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zhpr2_usm>(cgh, [=]() {
            ::cblas_zhpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void *)&alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
                     const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                     float *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ssbmv_usm>(cgh, [=]() {
            ::cblas_ssbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const float)alpha, a, (const int)lda, x, (const int)incx,
                          (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
                     const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                     double *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsbmv_usm>(cgh, [=]() {
            ::cblas_dsbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const double)alpha, a, (const int)lda, x, (const int)incx,
                          (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *ap, const float *x, int64_t incx, float beta, float *y,
                     int64_t incy, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sspmv_usm>(cgh, [=]() {
            ::cblas_sspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, ap, x, (const int)incx, (const float)beta, y,
                          (const int)incy);
        });
    });
    return done;
}

cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *ap, const double *x, int64_t incx, double beta, double *y,
                     int64_t incy, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dspmv_usm>(cgh, [=]() {
            ::cblas_dspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, ap, x, (const int)incx, (const double)beta, y,
                          (const int)incy);
        });
    });
    return done;
}

cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const float *x, int64_t incx, float *ap,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sspr_usm>(cgh, [=]() {
            ::cblas_sspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const double *x, int64_t incx, double *ap,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dspr_usm>(cgh, [=]() {
            ::cblas_dspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *x, int64_t incx, const float *y, int64_t incy, float *ap,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sspr2_usm>(cgh, [=]() {
            ::cblas_sspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *x, int64_t incx, const double *y, int64_t incy, double *ap,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dspr2_usm>(cgh, [=]() {
            ::cblas_dspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                     float *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ssymv_usm>(cgh, [=]() {
            ::cblas_ssymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, a, (const int)lda, x, (const int)incx,
                          (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                     double *y, int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsymv_usm>(cgh, [=]() {
            ::cblas_dsymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, a, (const int)lda, x, (const int)incx,
                          (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                    const float *x, int64_t incx, float *a, int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ssyr_usm>(cgh, [=]() {
            ::cblas_ssyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                    const double *x, int64_t incx, double *a, int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsyr_usm>(cgh, [=]() {
            ::cblas_dsyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, float alpha,
                     const float *x, int64_t incx, const float *y, int64_t incy, float *a,
                     int64_t lda, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ssyr2_usm>(cgh, [=]() {
            ::cblas_ssyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, int64_t n, double alpha,
                     const double *x, int64_t incx, const double *y, int64_t incy, double *a,
                     int64_t lda, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsyr2_usm>(cgh, [=]() {
            ::cblas_dsyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const float *a, int64_t lda, float *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_stbmv_usm>(cgh, [=]() {
            ::cblas_stbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const double *a, int64_t lda, double *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtbmv_usm>(cgh, [=]() {
            ::cblas_dtbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<float> *a, int64_t lda,
                     std::complex<float> *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctbmv_usm>(cgh, [=]() {
            ::cblas_ctbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<double> *a, int64_t lda,
                     std::complex<double> *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztbmv_usm>(cgh, [=]() {
            ::cblas_ztbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const float *a, int64_t lda, float *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_stbsv_usm>(cgh, [=]() {
            ::cblas_stbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const double *a, int64_t lda, double *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtbsv_usm>(cgh, [=]() {
            ::cblas_dtbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<float> *a, int64_t lda,
                     std::complex<float> *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctbsv_usm>(cgh, [=]() {
            ::cblas_ctbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, int64_t k, const std::complex<double> *a, int64_t lda,
                     std::complex<double> *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztbsv_usm>(cgh, [=]() {
            ::cblas_ztbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const float *ap, float *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_stpmv_usm>(cgh, [=]() {
            ::cblas_stpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const double *ap, double *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtpmv_usm>(cgh, [=]() {
            ::cblas_dtpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<float> *ap, std::complex<float> *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctpmv_usm>(cgh, [=]() {
            ::cblas_ctpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<double> *ap, std::complex<double> *x,
                     int64_t incx, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztpmv_usm>(cgh, [=]() {
            ::cblas_ztpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const float *ap, float *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_stpsv_usm>(cgh, [=]() {
            ::cblas_stpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const double *ap, double *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtpsv_usm>(cgh, [=]() {
            ::cblas_dtpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<float> *ap, std::complex<float> *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctpsv_usm>(cgh, [=]() {
            ::cblas_ctpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<double> *ap, std::complex<double> *x,
                     int64_t incx, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztpsv_usm>(cgh, [=]() {
            ::cblas_ztpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const float *a, int64_t lda, float *b, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_strmv_usm>(cgh, [=]() {
            ::cblas_strmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const double *a, int64_t lda, double *b, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtrmv_usm>(cgh, [=]() {
            ::cblas_dtrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const std::complex<float> *a, int64_t lda, std::complex<float> *b,
                     int64_t incx, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctrmv_usm>(cgh, [=]() {
            ::cblas_ctrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose transa, diag unit_diag,
                     int64_t n, const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                     int64_t incx, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztrmv_usm>(cgh, [=]() {
            ::cblas_ztrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const float *a, int64_t lda, float *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_strsv_usm>(cgh, [=]() {
            ::cblas_strsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const double *a, int64_t lda, double *x, int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtrsv_usm>(cgh, [=]() {
            ::cblas_dtrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<float> *a, int64_t lda, std::complex<float> *x,
                     int64_t incx, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctrsv_usm>(cgh, [=]() {
            ::cblas_ctrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     int64_t n, const std::complex<double> *a, int64_t lda, std::complex<double> *x,
                     int64_t incx, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztrsv_usm>(cgh, [=]() {
            ::cblas_ztrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}
