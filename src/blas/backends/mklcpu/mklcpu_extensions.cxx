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

template <typename Ta, typename Tb>
void gemm_bias_fallback(cl::sycl::queue &queue, MKL_LAYOUT layout, transpose transa,
                        transpose transb, offset offsetc, int64_t m, int64_t n, int64_t k,
                        float alpha, cl::sycl::buffer<Ta, 1> &a, int64_t lda, Ta ao,
                        cl::sycl::buffer<Tb, 1> &b, int64_t ldb, Tb bo, float beta,
                        cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
                        cl::sycl::buffer<int32_t, 1> &co) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto transa_ = cblas_convert(transa);
        auto transb_ = cblas_convert(transb);
        auto offsetc_ = cblas_convert(offsetc);
        int64_t sizea, sizeb, sizec;
        if (layout == MKL_COL_MAJOR) {
            sizea = (transa == transpose::nontrans) ? lda * k : lda * m;
            sizeb = (transb == transpose::nontrans) ? ldb * n : ldb * k;
            sizec = ldc * n;
        }
        else {
            sizea = (transa == transpose::nontrans) ? lda * m : lda * k;
            sizeb = (transb == transpose::nontrans) ? ldb * k : ldb * n;
            sizec = ldc * m;
        }
        auto accessor_a = a.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_co = co.template get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class mkl_kernel_gemm_bias_fallback>(cgh, [=]() {
            auto a_mat = (Ta *)static_cast<void *>(accessor_a.get_pointer());
            auto b_mat = (Tb *)static_cast<void *>(accessor_b.get_pointer());
            auto c_mat = (int32_t *)static_cast<void *>(accessor_c.get_pointer());
            auto co_vec = (int32_t *)static_cast<void *>(accessor_co.get_pointer());
            auto ad = (double *)malloc(sizeof(double) * sizea);
            auto bd = (double *)malloc(sizeof(double) * sizeb);
            auto cd = (double *)malloc(sizeof(double) * sizec);
            double alphad = alpha, betad = beta;
            double aod = ao, bod = bo;
            copy_mat(a_mat, layout, transa, m, k, lda, aod, ad);
            copy_mat(b_mat, layout, transb, k, n, ldb, bod, bd);
            copy_mat(c_mat, layout, MKL_NOTRANS, m, n, ldc, 0.0, cd);
            ::cblas_dgemm((CBLAS_LAYOUT)layout, transa_, transb_, m, n, k, alphad, ad, lda, bd, ldb,
                          betad, cd, ldc);
            copy_mat(cd, layout, m, n, ldc, offsetc_, co_vec, c_mat);
            free(ad);
            free(bd);
            free(cd);
        });
    });
}

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> &a,
               int64_t lda, int8_t ao, cl::sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb,
                       bo, beta, c, ldc, co);
}

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> &a,
               int64_t lda, int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    if (MKLMAJOR == MKL_COL_MAJOR && is_int8(-int(ao)) && is_int8(-int(bo))) {
        queue.submit([&](cl::sycl::handler &cgh) {
            CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
            CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
            CBLAS_OFFSET offsetc_ = cblas_convert(offsetc);
            auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
            auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
            auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto accessor_co = co.get_access<cl::sycl::access::mode::read>(cgh);
            host_task<class mkl_kernel_gemm_s8u8s32>(cgh, [=]() {
                MKL_INT8 *a_mat =
                    static_cast<MKL_INT8 *>(static_cast<void *>(accessor_a.get_pointer()));
                MKL_UINT8 *b_mat =
                    static_cast<MKL_UINT8 *>(static_cast<void *>(accessor_b.get_pointer()));
                MKL_INT8 bo_internal = -bo;
                MKL_INT8 ao_internal = -ao;
                ::cblas_gemm_s8u8s32(CBLASMAJOR, transa_, transb_, offsetc_, m, n, k, alpha, a_mat,
                                     lda, ao_internal, b_mat, ldb, bo_internal, beta,
                                     (MKL_INT32 *)accessor_c.get_pointer(), ldc,
                                     (const MKL_INT32 *)accessor_co.get_pointer());
            });
        });
    }
    else
        gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b,
                           ldb, bo, beta, c, ldc, co);
}

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<uint8_t, 1> &a,
               int64_t lda, uint8_t ao, cl::sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    if (MKLMAJOR == MKL_ROW_MAJOR && is_int8(-int(ao)) && is_int8(-int(bo))) {
        queue.submit([&](cl::sycl::handler &cgh) {
            CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
            CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
            CBLAS_OFFSET offsetc_ = cblas_convert(column_to_row(offsetc));
            auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
            auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
            auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto accessor_co = co.get_access<cl::sycl::access::mode::read>(cgh);
            host_task<class mkl_kernel_gemm_u8s8s32>(cgh, [=]() {
                MKL_UINT8 *a_mat =
                    static_cast<MKL_UINT8 *>(static_cast<void *>(accessor_a.get_pointer()));
                MKL_INT8 *b_mat =
                    static_cast<MKL_INT8 *>(static_cast<void *>(accessor_b.get_pointer()));
                MKL_INT8 bo_internal = -bo;
                MKL_INT8 ao_internal = -ao;
                ::cblas_gemm_s8u8s32(CblasColMajor, transb_, transa_, offsetc_, n, m, k, alpha,
                                     b_mat, ldb, bo_internal, a_mat, lda, ao_internal, beta,
                                     (MKL_INT32 *)accessor_c.get_pointer(), ldc,
                                     (const MKL_INT32 *)accessor_co.get_pointer());
            });
        });
    }
    else
        gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b,
                           ldb, bo, beta, c, ldc, co);
}

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<uint8_t, 1> &a,
               int64_t lda, uint8_t ao, cl::sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb,
                       bo, beta, c, ldc, co);
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
           int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_sgemmt>(cgh, [=]() {
            ::cblas_sgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, alpha,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb, beta,
                           accessor_c.get_pointer(), ldc);
        });
    });
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dgemmt>(cgh, [=]() {
            ::cblas_dgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, alpha,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb, beta,
                           accessor_c.get_pointer(), ldc);
        });
    });
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cgemmt>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, (const void *)&alpha_,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                           (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    queue.submit([&](cl::sycl::handler &cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        auto accessor_a = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zgemmt>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, (const void *)&alpha_,
                           accessor_a.get_pointer(), lda, accessor_b.get_pointer(), ldb,
                           (const void *)&beta_, accessor_c.get_pointer(), ldc);
        });
    });
}

// USM APIs

template <typename Ta, typename Tb>
cl::sycl::event gemm_bias_fallback(cl::sycl::queue &queue, MKL_LAYOUT layout, transpose transa,
                                   transpose transb, offset offsetc, int64_t m, int64_t n,
                                   int64_t k, float alpha, const Ta *a, int64_t lda, Ta ao,
                                   const Tb *b, int64_t ldb, Tb bo, float beta, int32_t *c,
                                   int64_t ldc, const int32_t *co,
                                   const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        auto transa_ = cblas_convert(transa);
        auto transb_ = cblas_convert(transb);
        auto offsetc_ = cblas_convert(offsetc);
        int64_t sizea, sizeb, sizec;
        if (layout == MKL_COL_MAJOR) {
            sizea = (transa == transpose::nontrans) ? lda * k : lda * m;
            sizeb = (transb == transpose::nontrans) ? ldb * n : ldb * k;
            sizec = ldc * n;
        }
        else {
            sizea = (transa == transpose::nontrans) ? lda * m : lda * k;
            sizeb = (transb == transpose::nontrans) ? ldb * k : ldb * n;
            sizec = ldc * m;
        }
        host_task<class mkl_kernel_gemm_bias_fallback_usm>(cgh, [=]() {
            double *ad = (double *)malloc(sizeof(double) * sizea);
            double *bd = (double *)malloc(sizeof(double) * sizeb);
            double *cd = (double *)malloc(sizeof(double) * sizec);
            double alphad = alpha, betad = beta;
            double aod = ao, bod = bo;
            copy_mat(a, layout, transa, m, k, lda, aod, ad);
            copy_mat(b, layout, transb, k, n, ldb, bod, bd);
            copy_mat(c, layout, MKL_NOTRANS, m, n, ldc, 0.0, cd);
            ::cblas_dgemm((CBLAS_LAYOUT)layout, transa_, transb_, m, n, k, alphad, ad, lda, bd, ldb,
                          betad, cd, ldc);
            copy_mat(cd, layout, m, n, ldc, offsetc_, co, c);
            free(ad);
            free(bd);
            free(cd);
        });
    });
    return done;
}

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const int8_t *a, int64_t lda, int8_t ao, const int8_t *b, int64_t ldb,
                          int8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda, ao,
                              b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const int8_t *a, int64_t lda, int8_t ao, const uint8_t *b, int64_t ldb,
                          uint8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    if (MKLMAJOR == MKL_COL_MAJOR && is_int8(-int(ao)) && is_int8(-int(bo))) {
        auto done = queue.submit([&](cl::sycl::handler &cgh) {
            int64_t num_events = dependencies.size();
            for (int64_t i = 0; i < num_events; i++) {
                cgh.depends_on(dependencies[i]);
            }
            CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
            CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
            CBLAS_OFFSET offsetc_ = cblas_convert(offsetc);
            host_task<class mkl_kernel_gemm_s8u8s32_usm>(cgh, [=]() {
                MKL_INT8 bo_internal = -bo;
                MKL_INT8 ao_internal = -ao;
                ::cblas_gemm_s8u8s32(CBLASMAJOR, transa_, transb_, offsetc_, m, n, k, alpha,
                                     (const MKL_INT8 *)a, lda, ao_internal, (const MKL_UINT8 *)b,
                                     ldb, bo_internal, beta, (MKL_INT32 *)c, ldc,
                                     (const MKL_INT32 *)co);
            });
        });
        return done;
    }
    else
        return gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda,
                                  ao, b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const uint8_t *a, int64_t lda, uint8_t ao, const int8_t *b, int64_t ldb,
                          int8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    if (MKLMAJOR == MKL_ROW_MAJOR && is_int8(-int(ao)) && is_int8(-int(bo))) {
        auto done = queue.submit([&](cl::sycl::handler &cgh) {
            int64_t num_events = dependencies.size();
            for (int64_t i = 0; i < num_events; i++) {
                cgh.depends_on(dependencies[i]);
            }
            CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
            CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
            CBLAS_OFFSET offsetc_ = cblas_convert(column_to_row(offsetc));
            host_task<class mkl_kernel_gemm_u8s8s32_usm>(cgh, [=]() {
                MKL_INT8 bo_internal = -bo;
                MKL_INT8 ao_internal = -ao;
                ::cblas_gemm_s8u8s32(CblasColMajor, transb_, transa_, offsetc_, n, m, k, alpha,
                                     (const MKL_INT8 *)b, ldb, bo_internal, (const MKL_UINT8 *)a,
                                     lda, ao_internal, beta, (MKL_INT32 *)c, ldc,
                                     (const MKL_INT32 *)co);
            });
        });
        return done;
    }
    else
        return gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda,
                                  ao, b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const uint8_t *a, int64_t lda, uint8_t ao, const uint8_t *b, int64_t ldb,
                          uint8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return gemm_bias_fallback(queue, MKLMAJOR, transa, transb, offsetc, m, n, k, alpha, a, lda, ao,
                              b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                      const float *b, int64_t ldb, float beta, float *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class mkl_kernel_sgemmt_usm>(cgh, [=]() {
            ::cblas_sgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, alpha, a, lda, b, ldb,
                           beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                      const double *b, int64_t ldb, double beta, double *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class mkl_kernel_dgemmt_usm>(cgh, [=]() {
            ::cblas_dgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, alpha, a, lda, b, ldb,
                           beta, c, ldc);
        });
    });
    return done;
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      int64_t lda, const std::complex<float> *b, int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_cgemmt_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_cgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, (const void *)&alpha_,
                           a, lda, b, ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                      int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zgemmt_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zgemmt(CBLASMAJOR, upper_lower_, transa_, transb_, n, k, (const void *)&alpha_,
                           a, lda, b, ldb, (const void *)&beta_, c, ldc);
        });
    });
    return done;
}
