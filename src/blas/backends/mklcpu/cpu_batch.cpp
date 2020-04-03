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

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<int64_t, 1> &k,
                cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                cl::sycl::buffer<int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                cl::sycl::buffer<int64_t, 1> &ldb, cl::sycl::buffer<float, 1> &beta,
                cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<int64_t, 1> &ldc,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>(cgh);
        auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto k_acc          = k.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc          = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);

        host_task<class mkl_kernel_init_sgemm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            float **a_array      = (float **)::malloc(sizeof(float *) * total_size);
            float **b_array      = (float **)::malloc(sizeof(float *) * total_size);
            float **c_array      = (float **)::malloc(sizeof(float *) * total_size);
            MKL_INT *m_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *k_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldc_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_ = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                k_[i]          = k_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                ldc_[i]        = ldc_acc[i];
                group_size_[i] = group_size_acc[i];
                transa_[i]     = *fortran_char(transa_acc[i]);
                transb_[i]     = *fortran_char(transb_acc[i]);

                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                        c_array[0] = c_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                        c_array[idx] = c_array[idx - 1] + offset_c;
                    }
                    idx++;
                    offset_a = (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i]
                                                                      : lda_acc[i] * m_acc[i];
                    offset_b = (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i]
                                                                      : ldb_acc[i] * k_acc[i];
                    offset_c = ldc_acc[i] * n_acc[i];
                }
            }

            ::sgemm_batch(transa_, transb_, m_, n_, k_, alpha_acc.get_pointer(),
                          (const float **)a_array, lda_, (const float **)b_array, ldb_,
                          beta_acc.get_pointer(), c_array, ldc_, (MKL_INT *)&group_count,
                          group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
            ::free(m_);
            ::free(n_);
            ::free(k_);
            ::free(lda_);
            ::free(ldb_);
            ::free(ldc_);
            ::free(group_size_);
            ::free(transa_);
            ::free(transb_);
        });
    });
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<int64_t, 1> &k,
                cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
                cl::sycl::buffer<int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
                cl::sycl::buffer<int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
                cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<int64_t, 1> &ldc,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>(cgh);
        auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto k_acc          = k.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc          = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);

        host_task<class mkl_kernel_dgemm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            double **a_array     = (double **)::malloc(sizeof(double *) * total_size);
            double **b_array     = (double **)::malloc(sizeof(double *) * total_size);
            double **c_array     = (double **)::malloc(sizeof(double *) * total_size);
            MKL_INT *m_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *k_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldc_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_ = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                k_[i]          = k_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                ldc_[i]        = ldc_acc[i];
                group_size_[i] = group_size_acc[i];
                transa_[i]     = *fortran_char(transa_acc[i]);
                transb_[i]     = *fortran_char(transb_acc[i]);

                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                        c_array[0] = c_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                        c_array[idx] = c_array[idx - 1] + offset_c;
                    }
                    idx++;
                    offset_a = (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i]
                                                                      : lda_acc[i] * m_acc[i];
                    offset_b = (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i]
                                                                      : ldb_acc[i] * k_acc[i];
                    offset_c = ldc_acc[i] * n_acc[i];
                }
            }

            ::dgemm_batch(transa_, transb_, m_, n_, k_, alpha_acc.get_pointer(),
                          (const double **)a_array, lda_, (const double **)b_array, ldb_,
                          beta_acc.get_pointer(), c_array, ldc_, (MKL_INT *)&group_count,
                          group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
            ::free(m_);
            ::free(n_);
            ::free(k_);
            ::free(lda_);
            ::free(ldb_);
            ::free(ldc_);
            ::free(group_size_);
            ::free(transa_);
            ::free(transb_);
        });
    });
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<int64_t, 1> &k,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<int64_t, 1> &ldb,
                cl::sycl::buffer<std::complex<float>, 1> &beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<int64_t, 1> &ldc,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>(cgh);
        auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto k_acc          = k.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc          = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);

        host_task<class mkl_kernel_cgemm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            MKL_Complex8 **a_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * total_size);
            MKL_Complex8 **b_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * total_size);
            MKL_Complex8 **c_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * total_size);
            MKL_INT *m_            = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_            = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *k_            = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldc_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_   = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                k_[i]          = k_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                ldc_[i]        = ldc_acc[i];
                group_size_[i] = group_size_acc[i];
                transa_[i]     = *fortran_char(transa_acc[i]);
                transb_[i]     = *fortran_char(transb_acc[i]);

                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                        c_array[0] = c_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                        c_array[idx] = c_array[idx - 1] + offset_c;
                    }
                    idx++;
                    offset_a = (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i]
                                                                      : lda_acc[i] * m_acc[i];
                    offset_b = (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i]
                                                                      : ldb_acc[i] * k_acc[i];
                    offset_c = ldc_acc[i] * n_acc[i];
                }
            }

            ::cgemm_batch(transa_, transb_, m_, n_, k_, alpha_acc.get_pointer(),
                          (const MKL_Complex8 **)a_array, lda_, (const MKL_Complex8 **)b_array,
                          ldb_, beta_acc.get_pointer(), c_array, ldc_, (MKL_INT *)&group_count,
                          group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
            ::free(m_);
            ::free(n_);
            ::free(k_);
            ::free(lda_);
            ::free(ldb_);
            ::free(ldc_);
            ::free(group_size_);
            ::free(transa_);
            ::free(transb_);
        });
    });
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<int64_t, 1> &k,
                cl::sycl::buffer<std::complex<double>, 1> &alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<int64_t, 1> &ldb,
                cl::sycl::buffer<std::complex<double>, 1> &beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<int64_t, 1> &ldc,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>(cgh);
        auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto k_acc          = k.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc          = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);

        host_task<class mkl_kernel_zgemm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            MKL_Complex16 **a_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * total_size);
            MKL_Complex16 **b_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * total_size);
            MKL_Complex16 **c_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * total_size);
            MKL_INT *m_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *k_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldc_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_ = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                k_[i]          = k_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                ldc_[i]        = ldc_acc[i];
                group_size_[i] = group_size_acc[i];
                transa_[i]     = *fortran_char(transa_acc[i]);
                transb_[i]     = *fortran_char(transb_acc[i]);

                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                        c_array[0] = c_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                        c_array[idx] = c_array[idx - 1] + offset_c;
                    }
                    idx++;
                    offset_a = (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i]
                                                                      : lda_acc[i] * m_acc[i];
                    offset_b = (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i]
                                                                      : ldb_acc[i] * k_acc[i];
                    offset_c = ldc_acc[i] * n_acc[i];
                }
            }

            ::zgemm_batch(transa_, transb_, m_, n_, k_, alpha_acc.get_pointer(),
                          (const MKL_Complex16 **)a_array, lda_, (const MKL_Complex16 **)b_array,
                          ldb_, beta_acc.get_pointer(), c_array, ldc_, (MKL_INT *)&group_count,
                          group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
            ::free(m_);
            ::free(n_);
            ::free(k_);
            ::free(lda_);
            ::free(ldb_);
            ::free(ldc_);
            ::free(group_size_);
            ::free(transa_);
            ::free(transb_);
        });
    });
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
                int64_t stride_a, cl::sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc   = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc   = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        char transa_ = *fortran_char(transa);
        char transb_ = *fortran_char(transb);
        MKL_INT one  = 1;

        host_task<class mkl_kernel_init_sgemm_batch_stride>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **b_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **c_array = (float **)::malloc(sizeof(float *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                    c_array[0] = c_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                    c_array[i] = c_array[i - 1] + stride_c;
                }
            }

            ::sgemm_batch(&transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          (const MKL_INT *)&k, &alpha, (const float **)a_array,
                          (const MKL_INT *)&lda, (const float **)b_array, (const MKL_INT *)&ldb,
                          &beta, c_array, (const MKL_INT *)&ldc, (const MKL_INT *)&one,
                          (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
                int64_t stride_a, cl::sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc   = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc   = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        char transa_ = *fortran_char(transa);
        char transb_ = *fortran_char(transb);
        MKL_INT one  = 1;

        host_task<class mkl_kernel_init_dgemm_batch_stride>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **b_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **c_array = (double **)::malloc(sizeof(double *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                    c_array[0] = c_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                    c_array[i] = c_array[i - 1] + stride_c;
                }
            }

            ::dgemm_batch(&transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          (const MKL_INT *)&k, &alpha, (const double **)a_array,
                          (const MKL_INT *)&lda, (const double **)b_array, (const MKL_INT *)&ldb,
                          &beta, c_array, (const MKL_INT *)&ldc, (const MKL_INT *)&one,
                          (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc   = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc   = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        char transa_ = *fortran_char(transa);
        char transb_ = *fortran_char(transb);
        MKL_INT one  = 1;

        host_task<class mkl_kernel_init_cgemm_batch_stride>(cgh, [=]() {
            MKL_Complex8 **a_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **b_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **c_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                    c_array[0] = c_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                    c_array[i] = c_array[i - 1] + stride_c;
                }
            }

            ::cgemm_batch(&transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          (const MKL_INT *)&k, &alpha, (const MKL_Complex8 **)a_array,
                          (const MKL_INT *)&lda, (const MKL_Complex8 **)b_array,
                          (const MKL_INT *)&ldb, &beta, c_array, (const MKL_INT *)&ldc,
                          (const MKL_INT *)&one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc   = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc   = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto c_acc   = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        char transa_ = *fortran_char(transa);
        char transb_ = *fortran_char(transb);
        MKL_INT one  = 1;

        host_task<class mkl_kernel_init_zgemm_batch_stride>(cgh, [=]() {
            MKL_Complex16 **a_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            MKL_Complex16 **b_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            MKL_Complex16 **c_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                    c_array[0] = c_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                    c_array[i] = c_array[i - 1] + stride_c;
                }
            }

            ::zgemm_batch(&transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          (const MKL_INT *)&k, &alpha, (const MKL_Complex16 **)a_array,
                          (const MKL_INT *)&lda, (const MKL_Complex16 **)b_array,
                          (const MKL_INT *)&ldb, &beta, c_array, (const MKL_INT *)&ldc,
                          (const MKL_INT *)&one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
                cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<int64_t, 1> &lda,
                cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<int64_t, 1> &ldb,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>(cgh);
        auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>(cgh);
        auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>(cgh);
        auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class mkl_kernel_init_strsm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            float **a_array      = (float **)::malloc(sizeof(float *) * total_size);
            float **b_array      = (float **)::malloc(sizeof(float *) * total_size);
            MKL_INT *m_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_ = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *side_  = (char *)::malloc(sizeof(char) * group_count);
            char *uplo_  = (char *)::malloc(sizeof(char) * group_count);
            char *trans_ = (char *)::malloc(sizeof(char) * group_count);
            char *diag_  = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                group_size_[i] = group_size_acc[i];
                trans_[i]      = *fortran_char(trans_acc[i]);
                side_[i]       = *fortran_char(side_acc[i]);
                uplo_[i]       = *fortran_char(uplo_acc[i]);
                diag_[i]       = *fortran_char(diag_acc[i]);

                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                    }
                    idx++;
                    offset_a =
                        (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
                    offset_b = ldb_acc[i] * n_acc[i];
                }
            }

            ::strsm_batch(side_, uplo_, trans_, diag_, m_, n_, alpha_acc.get_pointer(),
                          (const float **)a_array, lda_, (float **)b_array, ldb_,
                          (MKL_INT *)&group_count, group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(m_);
            ::free(n_);
            ::free(lda_);
            ::free(ldb_);
            ::free(group_size_);
            ::free(side_);
            ::free(uplo_);
            ::free(trans_);
            ::free(diag_);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<float, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc  = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc  = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        char trans_ = *fortran_char(trans);
        char side_  = *fortran_char(left_right);
        char uplo_  = *fortran_char(upper_lower);
        char diag_  = *fortran_char(unit_diag);
        MKL_INT one = 1;

        host_task<class mkl_kernel_init_strsm_batch_stride>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **b_array = (float **)::malloc(sizeof(float *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                }
            }

            ::strsm_batch(&side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          &alpha, (const float **)a_array, (const MKL_INT *)&lda, (float **)b_array,
                          (const MKL_INT *)&ldb, (const MKL_INT *)&one,
                          (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
                cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<int64_t, 1> &lda,
                cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<int64_t, 1> &ldb,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>(cgh);
        auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>(cgh);
        auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>(cgh);
        auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class mkl_kernel_init_dtrsm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            double **a_array     = (double **)::malloc(sizeof(double *) * total_size);
            double **b_array     = (double **)::malloc(sizeof(double *) * total_size);
            MKL_INT *m_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_ = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *side_  = (char *)::malloc(sizeof(char) * group_count);
            char *uplo_  = (char *)::malloc(sizeof(char) * group_count);
            char *trans_ = (char *)::malloc(sizeof(char) * group_count);
            char *diag_  = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                group_size_[i] = group_size_acc[i];
                trans_[i]      = *fortran_char(trans_acc[i]);
                side_[i]       = *fortran_char(side_acc[i]);
                uplo_[i]       = *fortran_char(uplo_acc[i]);
                diag_[i]       = *fortran_char(diag_acc[i]);

                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                    }
                    idx++;
                    offset_a =
                        (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
                    offset_b = ldb_acc[i] * n_acc[i];
                }
            }

            ::dtrsm_batch(side_, uplo_, trans_, diag_, m_, n_, alpha_acc.get_pointer(),
                          (const double **)a_array, lda_, (double **)b_array, ldb_,
                          (MKL_INT *)&group_count, group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(m_);
            ::free(n_);
            ::free(lda_);
            ::free(ldb_);
            ::free(group_size_);
            ::free(side_);
            ::free(uplo_);
            ::free(trans_);
            ::free(diag_);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                int64_t lda, int64_t stride_a, cl::sycl::buffer<double, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc  = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc  = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        char trans_ = *fortran_char(trans);
        char side_  = *fortran_char(left_right);
        char uplo_  = *fortran_char(upper_lower);
        char diag_  = *fortran_char(unit_diag);
        MKL_INT one = 1;

        host_task<class mkl_kernel_init_dtrsm_batch_stride>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **b_array = (double **)::malloc(sizeof(double *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                }
            }

            ::dtrsm_batch(&side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          &alpha, (const double **)a_array, (const MKL_INT *)&lda,
                          (double **)b_array, (const MKL_INT *)&ldb, (const MKL_INT *)&one,
                          (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<int64_t, 1> &ldb,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>(cgh);
        auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>(cgh);
        auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>(cgh);
        auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class mkl_kernel_init_ctrsm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            MKL_Complex8 **a_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * total_size);
            MKL_Complex8 **b_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * total_size);
            MKL_INT *m_            = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_            = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_   = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *side_  = (char *)::malloc(sizeof(char) * group_count);
            char *uplo_  = (char *)::malloc(sizeof(char) * group_count);
            char *trans_ = (char *)::malloc(sizeof(char) * group_count);
            char *diag_  = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                group_size_[i] = group_size_acc[i];
                trans_[i]      = *fortran_char(trans_acc[i]);
                side_[i]       = *fortran_char(side_acc[i]);
                uplo_[i]       = *fortran_char(uplo_acc[i]);
                diag_[i]       = *fortran_char(diag_acc[i]);

                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                    }
                    idx++;
                    offset_a =
                        (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
                    offset_b = ldb_acc[i] * n_acc[i];
                }
            }

            ::ctrsm_batch(side_, uplo_, trans_, diag_, m_, n_, alpha_acc.get_pointer(),
                          (const MKL_Complex8 **)a_array, lda_, (MKL_Complex8 **)b_array, ldb_,
                          (MKL_INT *)&group_count, group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(m_);
            ::free(n_);
            ::free(lda_);
            ::free(ldb_);
            ::free(group_size_);
            ::free(side_);
            ::free(uplo_);
            ::free(trans_);
            ::free(diag_);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc  = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc  = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        char trans_ = *fortran_char(trans);
        char side_  = *fortran_char(left_right);
        char uplo_  = *fortran_char(upper_lower);
        char diag_  = *fortran_char(unit_diag);
        MKL_INT one = 1;

        host_task<class mkl_kernel_init_ctrsm_batch_stride>(cgh, [=]() {
            MKL_Complex8 **a_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **b_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                }
            }

            ::ctrsm_batch(&side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          &alpha, (const MKL_Complex8 **)a_array, (const MKL_INT *)&lda,
                          (MKL_Complex8 **)b_array, (const MKL_INT *)&ldb, (const MKL_INT *)&one,
                          (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<int64_t, 1> &m,
                cl::sycl::buffer<int64_t, 1> &n, cl::sycl::buffer<std::complex<double>, 1> &alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<int64_t, 1> &ldb,
                int64_t group_count, cl::sycl::buffer<int64_t, 1> &group_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>(cgh);
        auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>(cgh);
        auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>(cgh);
        auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>(cgh);
        auto m_acc          = m.get_access<cl::sycl::access::mode::read>(cgh);
        auto n_acc          = n.get_access<cl::sycl::access::mode::read>(cgh);
        auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>(cgh);
        auto a_acc          = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc          = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>(cgh);
        auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>(cgh);

        host_task<class mkl_kernel_init_ztrsm_batch>(cgh, [=]() {
            int64_t total_size = 0;

            for (int64_t i = 0; i < group_count; i++) {
                total_size += group_size_acc[i];
            }

            MKL_Complex16 **a_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * total_size);
            MKL_Complex16 **b_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * total_size);
            MKL_INT *m_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *n_          = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *lda_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *ldb_        = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            MKL_INT *group_size_ = (MKL_INT *)::malloc(sizeof(MKL_INT) * group_count);
            int64_t offset_a = 0, offset_b = 0, offset_c = 0, idx = 0;
            char *side_  = (char *)::malloc(sizeof(char) * group_count);
            char *uplo_  = (char *)::malloc(sizeof(char) * group_count);
            char *trans_ = (char *)::malloc(sizeof(char) * group_count);
            char *diag_  = (char *)::malloc(sizeof(char) * group_count);

            for (int64_t i = 0; i < group_count; i++) {
                m_[i]          = m_acc[i];
                n_[i]          = n_acc[i];
                lda_[i]        = lda_acc[i];
                ldb_[i]        = ldb_acc[i];
                group_size_[i] = group_size_acc[i];
                trans_[i]      = *fortran_char(trans_acc[i]);
                side_[i]       = *fortran_char(side_acc[i]);
                uplo_[i]       = *fortran_char(uplo_acc[i]);
                diag_[i]       = *fortran_char(diag_acc[i]);
                for (int64_t j = 0; j < group_size_acc[i]; j++) {
                    if (idx == 0) {
                        a_array[0] = a_acc.get_pointer();
                        b_array[0] = b_acc.get_pointer();
                    }
                    else {
                        a_array[idx] = a_array[idx - 1] + offset_a;
                        b_array[idx] = b_array[idx - 1] + offset_b;
                    }
                    idx++;
                    offset_a =
                        (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
                    offset_b = ldb_acc[i] * n_acc[i];
                }
            }

            ::ztrsm_batch(side_, uplo_, trans_, diag_, m_, n_, alpha_acc.get_pointer(),
                          (const MKL_Complex16 **)a_array, lda_, (MKL_Complex16 **)b_array, ldb_,
                          (MKL_INT *)&group_count, group_size_);

            ::free(a_array);
            ::free(b_array);
            ::free(m_);
            ::free(n_);
            ::free(lda_);
            ::free(ldb_);
            ::free(group_size_);
            ::free(side_);
            ::free(uplo_);
            ::free(trans_);
            ::free(diag_);
        });
    });
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto a_acc  = a.get_access<cl::sycl::access::mode::read>(cgh);
        auto b_acc  = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        char trans_ = *fortran_char(trans);
        char side_  = *fortran_char(left_right);
        char uplo_  = *fortran_char(upper_lower);
        char diag_  = *fortran_char(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_init_ztrsm_batch_stride>(cgh, [=]() {
            MKL_Complex16 **a_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            MKL_Complex16 **b_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);

            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = a_acc.get_pointer();
                    b_array[0] = b_acc.get_pointer();
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                }
            }

            ::ztrsm_batch(&side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          &alpha, (const MKL_Complex16 **)a_array, (const MKL_INT *)&lda,
                          (MKL_Complex16 **)b_array, (const MKL_INT *)&ldb, (const MKL_INT *)&one,
                          (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

} // namespace mklcpu
} // namespace onemkl
