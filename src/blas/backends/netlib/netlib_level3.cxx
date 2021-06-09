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

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
          cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sgemm>(cgh, [=]() {
            ::cblas_sgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const float)alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                          (const int)ldb, (const float)beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
          int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dgemm>(cgh, [=]() {
            ::cblas_dgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const double)alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                          (const int)ldb, (const double)beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cgemm>(cgh, [=]() {
            ::cblas_cgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const void *)&alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                          (const int)ldb, (const void *)&beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zgemm>(cgh, [=]() {
            ::cblas_zgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const void *)&alpha,
                          accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                          (const int)ldb, (const void *)&beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, cl::sycl::half alpha,
          cl::sycl::buffer<cl::sycl::half, 1> &a, std::int64_t lda,
          cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb, cl::sycl::half beta,
          cl::sycl::buffer<cl::sycl::half, 1> &c, std::int64_t ldc) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm", "for row_major layout");
#endif
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<cl::sycl::half, 1> &a, int64_t lda,
          cl::sycl::buffer<cl::sycl::half, 1> &b, int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, int64_t ldc) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm", "for row_major layout");
#endif
}

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_chemm>(cgh, [=]() {
            ::cblas_chemm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb, (const void *)&beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zhemm>(cgh, [=]() {
            ::cblas_zhemm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb, (const void *)&beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, float beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cherk>(cgh, [=]() {
            ::cblas_cherk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const float)alpha, accessor_a.get_pointer(),
                          (const int)lda, (const float)beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, double beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zherk>(cgh, [=]() {
            ::cblas_zherk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const double)alpha, accessor_a.get_pointer(),
                          (const int)lda, (const double)beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cher2k>(cgh, [=]() {
            ::cblas_cher2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha,
                           accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                           (const int)ldb, (const float)beta, accessor_c.get_pointer(),
                           (const int)ldc);
        });
    });
}

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zher2k>(cgh, [=]() {
            ::cblas_zher2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha,
                           accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                           (const int)ldb, (const double)beta, accessor_c.get_pointer(),
                           (const int)ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
          int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ssymm>(cgh, [=]() {
            ::cblas_ssymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const float)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb, (const float)beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dsymm>(cgh, [=]() {
            ::cblas_dsymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const double)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb, (const double)beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_csymm>(cgh, [=]() {
            ::cblas_csymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb, (const void *)&beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zsymm>(cgh, [=]() {
            ::cblas_zsymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb, (const void *)&beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ssyrk>(cgh, [=]() {
            ::cblas_ssyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const float)alpha, accessor_a.get_pointer(),
                          (const int)lda, (const float)beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, double beta,
          cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dsyrk>(cgh, [=]() {
            ::cblas_dsyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const double)alpha, accessor_a.get_pointer(),
                          (const int)lda, (const double)beta, accessor_c.get_pointer(),
                          (const int)ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_csyrk>(cgh, [=]() {
            ::cblas_csyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const void *)&alpha,
                          accessor_a.get_pointer(), (const int)lda, (const void *)&beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zsyrk>(cgh, [=]() {
            ::cblas_zsyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const void *)&alpha,
                          accessor_a.get_pointer(), (const int)lda, (const void *)&beta,
                          accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
           int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ssyr2k>(cgh, [=]() {
            ::cblas_ssyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const float)alpha, accessor_a.get_pointer(),
                           (const int)lda, accessor_b.get_pointer(), (const int)ldb,
                           (const float)beta, accessor_c.get_pointer(), (const int)ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dsyr2k>(cgh, [=]() {
            ::cblas_dsyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const double)alpha,
                           accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                           (const int)ldb, (const double)beta, accessor_c.get_pointer(),
                           (const int)ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_csyr2k>(cgh, [=]() {
            ::cblas_csyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha,
                           accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                           (const int)ldb, (const void *)&beta, accessor_c.get_pointer(),
                           (const int)ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zsyr2k>(cgh, [=]() {
            ::cblas_zsyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha,
                           accessor_a.get_pointer(), (const int)lda, accessor_b.get_pointer(),
                           (const int)ldb, (const void *)&beta, accessor_c.get_pointer(),
                           (const int)ldc);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_strmm>(cgh, [=]() {
            ::cblas_strmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const float)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtrmm>(cgh, [=]() {
            ::cblas_dtrmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const double)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctrmm>(cgh, [=]() {
            ::cblas_ctrmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztrmm>(cgh, [=]() {
            ::cblas_ztrmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_strsm>(cgh, [=]() {
            ::cblas_strsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const float)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dtrsm>(cgh, [=]() {
            ::cblas_dtrsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const double)alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ctrsm>(cgh, [=]() {
            ::cblas_ctrsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ztrsm>(cgh, [=]() {
            ::cblas_ztrsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, accessor_a.get_pointer(), (const int)lda,
                          accessor_b.get_pointer(), (const int)ldb);
        });
    });
}

// USM APIs

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, float alpha, const float *a, int64_t lda, const float *b,
                     int64_t ldb, float beta, float *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sgemm_usm>(cgh, [=]() {
            ::cblas_sgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const float)alpha, a,
                          (const int)lda, b, (const int)ldb, (const float)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                     const double *b, int64_t ldb, double beta, double *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dgemm_usm>(cgh, [=]() {
            ::cblas_dgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const double)alpha, a,
                          (const int)lda, b, (const int)ldb, (const double)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *b, int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cgemm_usm>(cgh, [=]() {
            ::cblas_cgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const void *)&alpha, a,
                          (const int)lda, b, (const int)ldb, (const void *)&beta, c,
                          (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                     int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zgemm_usm>(cgh, [=]() {
            ::cblas_zgemm(MAJOR, convert_to_cblas_trans(transa), convert_to_cblas_trans(transb),
                          (const int)m, (const int)n, (const int)k, (const void *)&alpha, a,
                          (const int)lda, b, (const int)ldb, (const void *)&beta, c,
                          (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *b, int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_chemm_usm>(cgh, [=]() {
            ::cblas_chemm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb,
                          (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, const std::complex<double> *b, int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zhemm_usm>(cgh, [=]() {
            ::cblas_zhemm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb,
                          (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, float alpha, const std::complex<float> *a, int64_t lda, float beta,
                     std::complex<float> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cherk_usm>(cgh, [=]() {
            ::cblas_cherk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const float)alpha, a, (const int)lda,
                          (const float)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, double alpha, const std::complex<double> *a, int64_t lda,
                     double beta, std::complex<double> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zherk_usm>(cgh, [=]() {
            ::cblas_zherk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const double)alpha, a, (const int)lda,
                          (const double)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      int64_t lda, const std::complex<float> *b, int64_t ldb, float beta,
                      std::complex<float> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cher2k_usm>(cgh, [=]() {
            ::cblas_cher2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha, a, (const int)lda, b,
                           (const int)ldb, (const float)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      int64_t lda, const std::complex<double> *b, int64_t ldb, double beta,
                      std::complex<double> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zher2k_usm>(cgh, [=]() {
            ::cblas_zher2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha, a, (const int)lda, b,
                           (const int)ldb, (const double)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, float alpha, const float *a, int64_t lda, const float *b,
                     int64_t ldb, float beta, float *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ssymm_usm>(cgh, [=]() {
            ::cblas_ssymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const float)alpha, a, (const int)lda, b, (const int)ldb,
                          (const float)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, double alpha, const double *a, int64_t lda, const double *b,
                     int64_t ldb, double beta, double *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsymm_usm>(cgh, [=]() {
            ::cblas_dsymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const double)alpha, a, (const int)lda, b, (const int)ldb,
                          (const double)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *b, int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_csymm_usm>(cgh, [=]() {
            ::cblas_csymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb,
                          (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, const std::complex<double> *b, int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zsymm_usm>(cgh, [=]() {
            ::cblas_zsymm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb,
                          (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, float alpha, const float *a, int64_t lda, float beta, float *c,
                     int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ssyrk_usm>(cgh, [=]() {
            ::cblas_ssyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const float)alpha, a, (const int)lda,
                          (const float)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, double alpha, const double *a, int64_t lda, double beta, double *c,
                     int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsyrk_usm>(cgh, [=]() {
            ::cblas_dsyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const double)alpha, a, (const int)lda,
                          (const double)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_csyrk_usm>(cgh, [=]() {
            ::cblas_csyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const void *)&alpha, a, (const int)lda,
                          (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zsyrk_usm>(cgh, [=]() {
            ::cblas_zsyrk(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          (const int)n, (const int)k, (const void *)&alpha, a, (const int)lda,
                          (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, float alpha, const float *a, int64_t lda, const float *b,
                      int64_t ldb, float beta, float *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ssyr2k_usm>(cgh, [=]() {
            ::cblas_ssyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const float)alpha, a, (const int)lda, b,
                           (const int)ldb, (const float)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, double alpha, const double *a, int64_t lda, const double *b,
                      int64_t ldb, double beta, double *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsyr2k_usm>(cgh, [=]() {
            ::cblas_dsyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const double)alpha, a, (const int)lda, b,
                           (const int)ldb, (const double)beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      int64_t lda, const std::complex<float> *b, int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_csyr2k_usm>(cgh, [=]() {
            ::cblas_csyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha, a, (const int)lda, b,
                           (const int)ldb, (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      int64_t lda, const std::complex<double> *b, int64_t ldb,
                      std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zsyr2k_usm>(cgh, [=]() {
            ::cblas_zsyr2k(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                           (const int)n, (const int)k, (const void *)&alpha, a, (const int)lda, b,
                           (const int)ldb, (const void *)&beta, c, (const int)ldc);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, float alpha, const float *a, int64_t lda,
                     float *b, int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_strmm_usm>(cgh, [=]() {
            ::cblas_strmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const float)alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, double alpha, const double *a,
                     int64_t lda, double *b, int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtrmm_usm>(cgh, [=]() {
            ::cblas_dtrmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const double)alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, int64_t lda, std::complex<float> *b, int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctrmm_usm>(cgh, [=]() {
            ::cblas_ctrmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                     int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztrmm_usm>(cgh, [=]() {
            ::cblas_ztrmm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, float alpha, const float *a, int64_t lda,
                     float *b, int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_strsm_usm>(cgh, [=]() {
            ::cblas_strsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const float)alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, double alpha, const double *a,
                     int64_t lda, double *b, int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dtrsm_usm>(cgh, [=]() {
            ::cblas_dtrsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const double)alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, int64_t lda, std::complex<float> *b, int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ctrsm_usm>(cgh, [=]() {
            ::cblas_ctrsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                     int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ztrsm_usm>(cgh, [=]() {
            ::cblas_ztrsm(MAJOR, convert_to_cblas_side(left_right),
                          convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)m, (const int)n,
                          (const void *)&alpha, a, (const int)lda, b, (const int)ldb);
        });
    });
    return done;
}
