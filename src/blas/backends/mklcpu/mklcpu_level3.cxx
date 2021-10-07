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
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sgemm>(cgh, [=]() {
            ::cblas_sgemm(CBLASMAJOR, transa_, transb_, m, n, k, alpha, accessor_a.get_pointer(),
                          lda, accessor_b.get_pointer(), ldb, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
          int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dgemm>(cgh, [=]() {
            ::cblas_dgemm(CBLASMAJOR, transa_, transb_, m, n, k, alpha, accessor_a.get_pointer(),
                          lda, accessor_b.get_pointer(), ldb, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgemm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgemm(CBLASMAJOR, transa_, transb_, m, n, k, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                          (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgemm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgemm(CBLASMAJOR, transa_, transb_, m, n, k, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                          (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
          cl::sycl::buffer<half, 1> &b, int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c,
          int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        if (!verify_support<cl::sycl::half, cl::sycl::half>(queue, cl::sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "cl::sycl::half",
                "half is not supported by the device or the sycl compiler");
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float f32_alpha = (float)alpha;
        float f32_beta = (float)beta;
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_hgemm>(cgh, [=]() {
            int64_t sizea, sizeb, sizec;
#ifdef COLUMN_MAJOR
            sizea = (transa == transpose::N) ? lda * k : lda * m;
            sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
            sizec = ldc * n;
#endif
#ifdef ROW_MAJOR
            sizea = (transa == transpose::N) ? lda * m : lda * k;
            sizeb = (transb == transpose::N) ? ldb * k : ldb * n;
            sizec = ldc * m;
#endif
            // copy A, B and C to float
            float *f32_a = (float *)::malloc(sizeof(float) * sizea);
            float *f32_b = (float *)::malloc(sizeof(float) * sizeb);
            float *f32_c = (float *)::malloc(sizeof(float) * sizec);
            copy_mat(accessor_a, MKLMAJOR, transa, m, k, lda, 0.0f, f32_a);
            copy_mat(accessor_b, MKLMAJOR, transb, k, n, ldb, 0.0f, f32_b);
            copy_mat(accessor_c, MKLMAJOR, transpose::N, m, n, ldc, 0.0f, f32_c);
            ::cblas_sgemm(CBLASMAJOR, transa_, transb_, m, n, k, f32_alpha, f32_a, lda, f32_b, ldb,
                          f32_beta, f32_c, ldc);
            // copy C back to half
            half co = 0.0f;
            copy_mat(f32_c, MKLMAJOR, m, n, ldc, offset::F, &co, (half *)accessor_c.get_pointer());
            ::free(f32_a);
            ::free(f32_b);
            ::free(f32_c);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
          cl::sycl::buffer<half, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    if (!verify_support<cl::sycl::half, cl::sycl::half>(queue, cl::sycl::aspect::fp16)) {
        throw oneapi::mkl::unimplemented(
            "blas", "cl::sycl::half", "half is not supported by the device or the sycl compiler");
    }
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_gemm_f16f16f32>(cgh, [=]() {
            int64_t sizea, sizeb;
#ifdef COLUMN_MAJOR
            sizea = (transa == transpose::N) ? lda * k : lda * m;
            sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
#endif
#ifdef ROW_MAJOR
            sizea = (transa == transpose::N) ? lda * m : lda * k;
            sizeb = (transb == transpose::N) ? ldb * k : ldb * n;
#endif
            // copy A and B to float
            float *f32_a = (float *)::malloc(sizeof(float) * sizea);
            float *f32_b = (float *)::malloc(sizeof(float) * sizeb);
            copy_mat(accessor_a, MKLMAJOR, transa, m, k, lda, 0.0f, f32_a);
            copy_mat(accessor_b, MKLMAJOR, transb, k, n, ldb, 0.0f, f32_b);
            ::cblas_sgemm(CBLASMAJOR, transa_, transb_, m, n, k, alpha, f32_a, lda, f32_b, ldb,
                          beta, accessor_c.get_pointer(), ldc);
            ::free(f32_a);
            ::free(f32_b);
        });
    });
}

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<bfloat16, 1> &a, int64_t lda,
          cl::sycl::buffer<bfloat16, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_gemm_bf16bf16f32>(cgh, [=]() {
            auto a_mat = (MKL_BF16 *)static_cast<void *>(accessor_a.get_pointer());
            auto b_mat = (MKL_BF16 *)static_cast<void *>(accessor_b.get_pointer());
            ::cblas_gemm_bf16bf16f32(CBLASMAJOR, transa_, transb_, m, n, k, alpha, a_mat, lda,
                                     b_mat, ldb, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_chemm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chemm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                          (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zhemm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhemm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                          (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, float beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cherk>(cgh, [=]() {
            ::cblas_cherk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, accessor_a.get_pointer(),
                          lda, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, double beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zherk>(cgh, [=]() {
            ::cblas_zherk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, accessor_a.get_pointer(),
                          lda, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cher2k>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_cher2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb, beta,
                           accessor_c.get_pointer(), ldc);
        });
    });
}

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zher2k>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zher2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb, beta,
                           accessor_c.get_pointer(), ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
          int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssymm>(cgh, [=]() {
            ::cblas_ssymm(CBLASMAJOR, left_right_, upper_lower_, m, n, alpha,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb, beta,
                          accessor_c.get_pointer(), ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsymm>(cgh, [=]() {
            ::cblas_dsymm(CBLASMAJOR, left_right_, upper_lower_, m, n, alpha,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb, beta,
                          accessor_c.get_pointer(), ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_csymm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csymm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                          (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zsymm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsymm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                          (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssyrk>(cgh, [=]() {
            ::cblas_ssyrk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, accessor_a.get_pointer(),
                          lda, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, double beta,
          cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsyrk>(cgh, [=]() {
            ::cblas_dsyrk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, accessor_a.get_pointer(),
                          lda, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_csyrk>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csyrk(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, (const void *)&beta_,
                          accessor_c.get_pointer(), ldc);
        });
    });
}

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zsyrk>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsyrk(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_,
                          accessor_a.get_pointer(), lda, (const void *)&beta_,
                          accessor_c.get_pointer(), ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
           int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ssyr2k>(cgh, [=]() {
            ::cblas_ssyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, accessor_a.get_pointer(),
                           lda, accessor_b.get_pointer(), ldb, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dsyr2k>(cgh, [=]() {
            ::cblas_dsyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, accessor_a.get_pointer(),
                           lda, accessor_b.get_pointer(), ldb, beta, accessor_c.get_pointer(), ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_csyr2k>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                           (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zsyr2k>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                           (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_strmm>(cgh, [=]() {
            ::cblas_strmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtrmm>(cgh, [=]() {
            ::cblas_dtrmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctrmm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ctrmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, accessor_a.get_pointer(), lda,
                          accessor_b.get_pointer(), ldb);
        });
    });
}

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztrmm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ztrmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, accessor_a.get_pointer(), lda,
                          accessor_b.get_pointer(), ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_strsm>(cgh, [=]() {
            ::cblas_strsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dtrsm>(cgh, [=]() {
            ::cblas_dtrsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ctrsm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ctrsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, accessor_a.get_pointer(), lda,
                          accessor_b.get_pointer(), ldb);
        });
    });
}

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ztrsm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ztrsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, accessor_a.get_pointer(), lda,
                          accessor_b.get_pointer(), ldb);
        });
    });
}

// USM APIs

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, float alpha, const float *a, int64_t lda, const float *b,
                     int64_t ldb, float beta, float *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class mkl_kernel_sgemm_usm>(cgh, [=]() {
            ::cblas_sgemm(CBLASMAJOR, transa_, transb_, m, n, k, alpha, a, lda, b, ldb, beta, c,
                          ldc);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                     const double *b, int64_t ldb, double beta, double *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class mkl_kernel_dgemm_usm>(cgh, [=]() {
            ::cblas_dgemm(CBLASMAJOR, transa_, transb_, m, n, k, alpha, a, lda, b, ldb, beta, c,
                          ldc);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *b, int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_cgemm_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgemm(CBLASMAJOR, transa_, transb_, m, n, k, (const void *)&alpha_, a, lda, b,
                          ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                     int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zgemm_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgemm(CBLASMAJOR, transa_, transb_, m, n, k, (const void *)&alpha_, a, lda, b,
                          ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, half alpha, const half *a, int64_t lda, const half *b,
                     int64_t ldb, half beta, half *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        if (!verify_support<cl::sycl::half, cl::sycl::half>(queue, cl::sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "cl::sycl::half",
                "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float f32_alpha = (float)alpha;
        float f32_beta = (float)beta;
        host_task<class mkl_kernel_hgemm_usm>(cgh, [=]() {
            int64_t sizea, sizeb, sizec;
#ifdef COLUMN_MAJOR
            sizea = (transa == transpose::N) ? lda * k : lda * m;
            sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
            sizec = ldc * n;
#endif
#ifdef ROW_MAJOR
            sizea = (transa == transpose::N) ? lda * m : lda * k;
            sizeb = (transb == transpose::N) ? ldb * k : ldb * n;
            sizec = ldc * m;
#endif
            // copy A, B and C to float
            float *f32_a = (float *)::malloc(sizeof(float) * sizea);
            float *f32_b = (float *)::malloc(sizeof(float) * sizeb);
            float *f32_c = (float *)::malloc(sizeof(float) * sizec);
            copy_mat(a, MKLMAJOR, transa, m, k, lda, 0.0f, f32_a);
            copy_mat(b, MKLMAJOR, transb, k, n, ldb, 0.0f, f32_b);
            copy_mat(c, MKLMAJOR, transpose::N, m, n, ldc, 0.0f, f32_c);
            ::cblas_sgemm(CBLASMAJOR, transa_, transb_, m, n, k, f32_alpha, f32_a, lda, f32_b, ldb,
                          f32_beta, f32_c, ldc);
            // copy C back to half
            half co = 0.0f;
            copy_mat(f32_c, MKLMAJOR, m, n, ldc, offset::F, &co, c);
            ::free(f32_a);
            ::free(f32_b);
            ::free(f32_c);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, float alpha, const half *a, int64_t lda, const half *b,
                     int64_t ldb, float beta, float *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        if (!verify_support<cl::sycl::half, cl::sycl::half>(queue, cl::sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "cl::sycl::half",
                "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float f32_alpha = (float)alpha;
        float f32_beta = (float)beta;
        host_task<class mkl_kernel_gemm_f16f16f32_usm>(cgh, [=]() {
            int64_t sizea, sizeb, sizec;
#ifdef COLUMN_MAJOR
            sizea = (transa == transpose::N) ? lda * k : lda * m;
            sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
#endif
#ifdef ROW_MAJOR
            sizea = (transa == transpose::N) ? lda * m : lda * k;
            sizeb = (transb == transpose::N) ? ldb * k : ldb * n;
#endif
            // copy A, B to float
            float *f32_a = (float *)::malloc(sizeof(float) * sizea);
            float *f32_b = (float *)::malloc(sizeof(float) * sizeb);
            copy_mat(a, MKLMAJOR, transa, m, k, lda, 0.0f, f32_a);
            copy_mat(b, MKLMAJOR, transb, k, n, ldb, 0.0f, f32_b);
            ::cblas_sgemm(CBLASMAJOR, transa_, transb_, m, n, k, f32_alpha, f32_a, lda, f32_b, ldb,
                          f32_beta, c, ldc);
            ::free(f32_a);
            ::free(f32_b);
        });
    });
    return done;
}

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                     int64_t n, int64_t k, float alpha, const bfloat16 *a, int64_t lda,
                     const bfloat16 *b, int64_t ldb, float beta, float *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class mkl_kernel_gemm_bf16bf16f32_usm>(cgh, [=]() {
            ::cblas_gemm_bf16bf16f32(CBLASMAJOR, transa_, transb_, m, n, k, alpha, (MKL_BF16 *)a,
                                     lda, (MKL_BF16 *)b, ldb, beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *b, int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_chemm_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_chemm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_, a,
                          lda, b, ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, const std::complex<double> *b, int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zhemm_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zhemm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_, a,
                          lda, b, ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, float alpha, const std::complex<float> *a, int64_t lda, float beta,
                     std::complex<float> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_cherk_usm>(cgh, [=]() {
            ::cblas_cherk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, a, lda, beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, double alpha, const std::complex<double> *a, int64_t lda,
                     double beta, std::complex<double> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_zherk_usm>(cgh, [=]() {
            ::cblas_zherk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, a, lda, beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      int64_t lda, const std::complex<float> *b, int64_t ldb, float beta,
                      std::complex<float> *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_cher2k_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_cher2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_, a, lda, b,
                           ldb, beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      int64_t lda, const std::complex<double> *b, int64_t ldb, double beta,
                      std::complex<double> *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_zher2k_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_zher2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_, a, lda, b,
                           ldb, beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, float alpha, const float *a, int64_t lda, const float *b,
                     int64_t ldb, float beta, float *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_ssymm_usm>(cgh, [=]() {
            ::cblas_ssymm(CBLASMAJOR, left_right_, upper_lower_, m, n, alpha, a, lda, b, ldb, beta,
                          c, ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, double alpha, const double *a, int64_t lda, const double *b,
                     int64_t ldb, double beta, double *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dsymm_usm>(cgh, [=]() {
            ::cblas_dsymm(CBLASMAJOR, left_right_, upper_lower_, m, n, alpha, a, lda, b, ldb, beta,
                          c, ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, const std::complex<float> *b, int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_csymm_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csymm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_, a,
                          lda, b, ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, int64_t m,
                     int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, const std::complex<double> *b, int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zsymm_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsymm(CBLASMAJOR, left_right_, upper_lower_, m, n, (const void *)&alpha_, a,
                          lda, b, ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, float alpha, const float *a, int64_t lda, float beta, float *c,
                     int64_t ldc, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_ssyrk_usm>(cgh, [=]() {
            ::cblas_ssyrk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, a, lda, beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, double alpha, const double *a, int64_t lda, double beta, double *c,
                     int64_t ldc, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_dsyrk_usm>(cgh, [=]() {
            ::cblas_dsyrk(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, a, lda, beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     int64_t lda, std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_csyrk_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csyrk(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_, a, lda,
                          (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                     int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                     int64_t lda, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zsyrk_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsyrk(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_, a, lda,
                          (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, float alpha, const float *a, int64_t lda, const float *b,
                      int64_t ldb, float beta, float *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_ssyr2k_usm>(cgh, [=]() {
            ::cblas_ssyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, a, lda, b, ldb, beta, c,
                           ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, double alpha, const double *a, int64_t lda, const double *b,
                      int64_t ldb, double beta, double *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_dsyr2k_usm>(cgh, [=]() {
            ::cblas_dsyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, alpha, a, lda, b, ldb, beta, c,
                           ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      int64_t lda, const std::complex<float> *b, int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_csyr2k_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_, a, lda, b,
                           ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                      int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      int64_t lda, const std::complex<double> *b, int64_t ldb,
                      std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zsyr2k_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsyr2k(CBLASMAJOR, upper_lower_, trans_, n, k, (const void *)&alpha_, a, lda, b,
                           ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, float alpha, const float *a, int64_t lda,
                     float *b, int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_strmm_usm>(cgh, [=]() {
            ::cblas_strmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          a, lda, b, ldb);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, double alpha, const double *a,
                     int64_t lda, double *b, int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtrmm_usm>(cgh, [=]() {
            ::cblas_dtrmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          a, lda, b, ldb);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, int64_t lda, std::complex<float> *b, int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_ctrmm_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ctrmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, a, lda, b, ldb);
        });
    });
    return done;
}

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                     int64_t ldb, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_ztrmm_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ztrmm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, a, lda, b, ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, float alpha, const float *a, int64_t lda,
                     float *b, int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_strsm_usm>(cgh, [=]() {
            ::cblas_strsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          a, lda, b, ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, double alpha, const double *a,
                     int64_t lda, double *b, int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class mkl_kernel_dtrsm_usm>(cgh, [=]() {
            ::cblas_dtrsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n, alpha,
                          a, lda, b, ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, int64_t lda, std::complex<float> *b, int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_ctrsm_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ctrsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, a, lda, b, ldb);
        });
    });
    return done;
}

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                     int64_t ldb, const std::vector<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_ztrsm_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::cblas_ztrsm(CBLASMAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                          (const void *)&alpha_, a, lda, b, ldb);
        });
    });
    return done;
}
