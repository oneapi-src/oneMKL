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

#ifndef _ONEMKL_BLAS_LOADER_HPP_
#define _ONEMKL_BLAS_LOADER_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

#include <onemkl/types.hpp>

namespace onemkl {
namespace blas {
namespace detail {

void herk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc);
void herk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc);

void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx);
void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx);
void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);
void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void scal(char *libname, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx);
void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx);
void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void trmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx);
void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx);
void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void tpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void spr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a);
void spr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a);

void gemm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<float, 1> &beta,
                cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
void gemm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
                cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
void gemm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                cl::sycl::buffer<std::complex<float>, 1> &beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
void gemm_batch(
    char *libname, cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size);
void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);
void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);
void gemm_batch(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc);
void syrk(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc);

void her2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
void her2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void hbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void hbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void rot(char *libname, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c, float s);
void rot(char *libname, cl::sycl::queue &queue, std::int64_t n,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c, double s);
void rot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
         std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s);
void rot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
         std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s);

void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy);
void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy);
void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void axpy(char *libname, cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

void gerc(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
void gerc(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
           cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
           cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
void syr2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc);

void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy);
void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy);
void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void gemv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void her(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
void her(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void hpr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a);
void hpr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a);

void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
              cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);
void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
              cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
              std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
              std::int64_t ldc);
void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
              std::int64_t ldc);
void gemm_ext(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c, std::int64_t ldc);

void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result);
void iamin(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result);

void hpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void hpmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void spmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
void spmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void rotmg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
           cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
           cl::sycl::buffer<float, 1> &param);
void rotmg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
           cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1, double y1,
           cl::sycl::buffer<double, 1> &param);

void swap(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
void swap(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
void swap(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void swap(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

void geru(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);
void geru(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result);
void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result);
void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result);
void nrm2(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result);

void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
void gemmt(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose transa,
           transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc);

void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);
void gemm(char *libname, cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc);

void syr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda);
void syr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda);

void ger(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda);
void ger(char *libname, cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda);

void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb);
void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb);
void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
void trsm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

void dotu(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result);
void dotu(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result);

void hemm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
void hemm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc);

void hpr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a);
void hpr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a);

void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy);
void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy);
void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void gbmv(char *libname, cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx);
void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx);
void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void tbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);
void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc);
void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
void symm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc);

void dotc(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result);
void dotc(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result);

void syr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda);
void syr(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda);

void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb);
void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb);
void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);
void trmm(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

void symv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
void symv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx);
void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx);
void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void tpsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx);
void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx);
void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void trsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void copy(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);
void copy(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);
void copy(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void copy(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

void hemv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);
void hemv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);
void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result);
void iamax(char *libname, cl::sycl::queue &queue, std::int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result);

void sbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy);
void sbmv(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void asum(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result);
void asum(char *libname, cl::sycl::queue &queue, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result);
void asum(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result);
void asum(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result);

void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx);
void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx);
void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);
void tbsv(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void spr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a);
void spr2(char *libname, cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a);

void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
                cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
                cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size);
void trsm_batch(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
                cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::complex<double>, 1> &alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a,
                cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<double>, 1> &b,
                cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
                cl::sycl::buffer<std::int64_t, 1> &group_size);
void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);
void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);
void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size);
void trsm_batch(char *libname, cl::sycl::queue &queue, side left_right, uplo upper_lower,
                transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size);

void rotm(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
          cl::sycl::buffer<float, 1> &param);
void rotm(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
          cl::sycl::buffer<double, 1> &param);

void dot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
         std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
         cl::sycl::buffer<float, 1> &result);
void dot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
         std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
         cl::sycl::buffer<double, 1> &result);
void dot(char *libname, cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
         std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
         cl::sycl::buffer<double, 1> &result);

void sdsdot(char *libname, cl::sycl::queue &queue, std::int64_t n, float sb,
            cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
            std::int64_t incy, cl::sycl::buffer<float, 1> &result);

void her2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           float beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);
void her2k(char *libname, cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           double beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<float, 1> &s);
void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<double, 1> &s);
void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s);
void rotg(char *libname, cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s);
} //namespace detail
} //namespace blas
} //namespace onemkl

#endif //_ONEMKL_BLAS_LOADER_HPP_
