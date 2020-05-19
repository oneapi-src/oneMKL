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
#include "onemkl/blas/detail/mklcpu/onemkl_blas_mklcpu.hpp"

namespace onemkl {
namespace mklcpu {

// Buffer APIs

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
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }

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
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }

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
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }

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
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }

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
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }

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
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }

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
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }

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
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }

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

// USM APIs

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, float *alpha, const float **a, int64_t *lda,
                           const float **b, int64_t *ldb, float *beta, float **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_usm_sgemm>(cgh, [=]() {
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = *fortran_char(transa[i]);
                transb_[i] = *fortran_char(transb[i]);
            }
            ::sgemm_batch(transa_, transb_, (const MKL_INT *)m, (const MKL_INT *)n,
                          (const MKL_INT *)k, alpha, (const float **)a, (const MKL_INT *)lda,
                          (const float **)b, (const MKL_INT *)ldb, beta, c, (const MKL_INT *)ldc,
                          (const MKL_INT *)&group_count, (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, double *alpha, const double **a, int64_t *lda,
                           const double **b, int64_t *ldb, double *beta, double **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_dgemm_batch_usm>(cgh, [=]() {
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = *fortran_char(transa[i]);
                transb_[i] = *fortran_char(transb[i]);
            }
            ::dgemm_batch(transa_, transb_, (const MKL_INT *)m, (const MKL_INT *)n,
                          (const MKL_INT *)k, alpha, (const double **)a, (const MKL_INT *)lda,
                          (const double **)b, (const MKL_INT *)ldb, beta, c, (const MKL_INT *)ldc,
                          (const MKL_INT *)&group_count, (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<float> *alpha,
                           const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **b, int64_t *ldb, std::complex<float> *beta,
                           std::complex<float> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_cgemm_batch_usm>(cgh, [=]() {
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = *fortran_char(transa[i]);
                transb_[i] = *fortran_char(transb[i]);
            }
            ::cgemm_batch(transa_, transb_, (const MKL_INT *)m, (const MKL_INT *)n,
                          (const MKL_INT *)k, alpha, (const std::complex<float> **)a,
                          (const MKL_INT *)lda, (const std::complex<float> **)b,
                          (const MKL_INT *)ldb, beta, c, (const MKL_INT *)ldc,
                          (const MKL_INT *)&group_count, (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<double> *alpha,
                           const std::complex<double> **a, int64_t *lda,
                           const std::complex<double> **b, int64_t *ldb, std::complex<double> *beta,
                           std::complex<double> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zgemm_batch_usm>(cgh, [=]() {
            char *transa_ = (char *)::malloc(sizeof(char) * group_count);
            char *transb_ = (char *)::malloc(sizeof(char) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = *fortran_char(transa[i]);
                transb_[i] = *fortran_char(transb[i]);
            }
            ::zgemm_batch(transa_, transb_, (const MKL_INT *)m, (const MKL_INT *)n,
                          (const MKL_INT *)k, alpha, (const std::complex<double> **)a,
                          (const MKL_INT *)lda, (const std::complex<double> **)b,
                          (const MKL_INT *)ldb, beta, c, (const MKL_INT *)ldc,
                          (const MKL_INT *)&group_count, (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                           int64_t stride_a, const float *b, int64_t ldb, int64_t stride_b,
                           float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        MKL_INT one        = 1;
        host_task<class mkl_kernel_sgemm_batch_usm>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **b_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **c_array = (float **)::malloc(sizeof(float *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = (float *)a;
                    b_array[0] = (float *)b;
                    c_array[0] = (float *)c;
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
    return done;
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                           int64_t stride_a, const double *b, int64_t ldb, int64_t stride_b,
                           double beta, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        MKL_INT one        = 1;
        host_task<class mkl_kernel_dgemm_batch_usm>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **b_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **c_array = (double **)::malloc(sizeof(double *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = (double *)a;
                    b_array[0] = (double *)b;
                    c_array[0] = (double *)c;
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
    return done;
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, int64_t stride_a,
                           const std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        MKL_INT one        = 1;
        host_task<class mkl_kernel_cgemm_batch_usm>(cgh, [=]() {
            std::complex<float> **a_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            std::complex<float> **b_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            std::complex<float> **c_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = (std::complex<float> *)a;
                    b_array[0] = (std::complex<float> *)b;
                    c_array[0] = (std::complex<float> *)c;
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                    c_array[i] = c_array[i - 1] + stride_c;
                }
            }
            ::cgemm_batch(&transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          (const MKL_INT *)&k, &alpha, (const std::complex<float> **)a_array,
                          (const MKL_INT *)&lda, (const std::complex<float> **)b_array,
                          (const MKL_INT *)&ldb, &beta, c_array, (const MKL_INT *)&ldc,
                          (const MKL_INT *)&one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
    return done;
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda, int64_t stride_a,
                           const std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        const char transa_ = *fortran_char(transa);
        const char transb_ = *fortran_char(transb);
        MKL_INT one        = 1;
        host_task<class mkl_kernel_zgemm_batch_usm>(cgh, [=]() {
            std::complex<double> **a_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            std::complex<double> **b_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            std::complex<double> **c_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                if (i == 0) {
                    a_array[0] = (std::complex<double> *)a;
                    b_array[0] = (std::complex<double> *)b;
                    c_array[0] = (std::complex<double> *)c;
                }
                else {
                    a_array[i] = a_array[i - 1] + stride_a;
                    b_array[i] = b_array[i - 1] + stride_b;
                    c_array[i] = c_array[i - 1] + stride_c;
                }
            }
            ::zgemm_batch(&transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                          (const MKL_INT *)&k, &alpha, (const std::complex<double> **)a_array,
                          (const MKL_INT *)&lda, (const std::complex<double> **)b_array,
                          (const MKL_INT *)&ldb, &beta, c_array, (const MKL_INT *)&ldc,
                          (const MKL_INT *)&one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
    return done;
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_saxpy_batch_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (int64_t i = 0; i < group_count; i++) {
                for (int64_t j = 0; j < group_size[i]; j++) {
                    ::saxpy((const MKL_INT *)(n + i), (const float *)(alpha + i), x[offset + j],
                            (const MKL_INT *)(incx + i), y[offset + j],
                            (const MKL_INT *)(incy + i));
                }
                offset += group_size[i];
            }
        });
    });
    return done;
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_daxpy_batch_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (int64_t i = 0; i < group_count; i++) {
                for (int64_t j = 0; j < group_size[i]; j++) {
                    ::daxpy((const MKL_INT *)(n + i), (const double *)(alpha + i), x[offset + j],
                            (const MKL_INT *)(incx + i), y[offset + j],
                            (const MKL_INT *)(incy + i));
                }
                offset += group_size[i];
            }
        });
    });
    return done;
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_caxpy_batch_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (int64_t i = 0; i < group_count; i++) {
                for (int64_t j = 0; j < group_size[i]; j++) {
                    MKL_Complex8 alpha_ = { alpha[i].real(), alpha[i].imag() };
                    ::caxpy((const MKL_INT *)(n + i), (const MKL_Complex8 *)&alpha_, x[offset + j],
                            (const MKL_INT *)(incx + i), y[offset + j],
                            (const MKL_INT *)(incy + i));
                }
                offset += group_size[i];
            }
        });
    });
    return done;
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zaxpy_batch_usm>(cgh, [=]() {
            int64_t offset = 0;
            for (int64_t i = 0; i < group_count; i++) {
                for (int64_t j = 0; j < group_size[i]; j++) {
                    MKL_Complex16 alpha_ = { alpha[i].real(), alpha[i].imag() };
                    ::zaxpy((const MKL_INT *)(n + i), (const MKL_Complex16 *)&alpha_, x[offset + j],
                            (const MKL_INT *)(incx + i), y[offset + j],
                            (const MKL_INT *)(incy + i));
                }
                offset += group_size[i];
            }
        });
    });
    return done;
}

} // namespace mklcpu
} // namespace onemkl
