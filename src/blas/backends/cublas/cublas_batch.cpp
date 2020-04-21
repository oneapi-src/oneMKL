/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#include <stdexcept>
#include "onemkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

namespace onemkl {
namespace cublas {

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<float, 1> &beta,
                cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
                cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                cl::sycl::buffer<std::complex<float>, 1> &beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, float beta, cl::sycl::buffer<float, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, double beta,
                cl::sycl::buffer<double, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
                cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
                cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::complex<double>, 1> &alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<double>, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
                cl::sycl::buffer<std::int64_t, 1> &group_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw std::runtime_error("Not implemented for cublas");
}
} // namespace cublas
} // namespace onemkl
