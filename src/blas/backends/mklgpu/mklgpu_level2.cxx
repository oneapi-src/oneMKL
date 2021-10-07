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

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::sgemv(queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::dgemv(queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::cgemv(queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::oneapi::mkl::gpu::zgemv(queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::sgbmv(queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                              incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::dgbmv(queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                              incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    ::oneapi::mkl::gpu::cgbmv(queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                              incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::oneapi::mkl::gpu::zgbmv(queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                              incx, beta, y, incy);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::sger(queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::dger(queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::cgerc(queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::zgerc(queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::cgeru(queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::zgeru(queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::chbmv(queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void hbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::oneapi::mkl::gpu::zhbmv(queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void hemv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::chemv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x, incx, beta, y,
                              incy);
}

void hemv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::oneapi::mkl::gpu::zhemv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x, incx, beta, y,
                              incy);
}

void her(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::cher(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a, lda);
}

void her(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::zher(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a, lda);
}

void her2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::cher2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a, lda);
}

void her2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::zher2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a, lda);
}

void hpmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    ::oneapi::mkl::gpu::chpmv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void hpmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::oneapi::mkl::gpu::zhpmv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void hpr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    ::oneapi::mkl::gpu::chpr(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a);
}

void hpr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    ::oneapi::mkl::gpu::zhpr(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a);
}

void hpr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    ::oneapi::mkl::gpu::chpr2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a);
}

void hpr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    ::oneapi::mkl::gpu::zhpr2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a);
}

void sbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::ssbmv(queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void sbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::dsbmv(queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x, incx, beta,
                              y, incy);
}

void spmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::sspmv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::dspmv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void spr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a) {
    ::oneapi::mkl::gpu::sspr(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a);
}

void spr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a) {
    ::oneapi::mkl::gpu::dspr(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a);
}

void spr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a) {
    ::oneapi::mkl::gpu::sspr2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a);
}

void spr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a) {
    ::oneapi::mkl::gpu::dspr2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a);
}

void symv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::ssymv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x, incx, beta, y,
                              incy);
}

void symv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::oneapi::mkl::gpu::dsymv(queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x, incx, beta, y,
                              incy);
}

void syr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    ::oneapi::mkl::gpu::ssyr(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a, lda);
}

void syr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    ::oneapi::mkl::gpu::dsyr(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a, lda);
}

void syr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::ssyr2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a, lda);
}

void syr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ::oneapi::mkl::gpu::dsyr2(queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y, incy, a, lda);
}

void tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::stbmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::dtbmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ctbmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ztbmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::stbsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::dtbsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ctbsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ztbsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, k, a, lda, x, incx);
}

void tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::stpmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::dtpmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::ctpmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ztpmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::stpsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::dtpsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::ctpsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ztpsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, x, incx);
}

void trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::strmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::dtrmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ctrmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ztrmv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::strsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    ::oneapi::mkl::gpu::dtrsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ctrsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::oneapi::mkl::gpu::ztrsv(queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                              mkl_convert(diag), n, a, lda, x, incx);
}

// USM APIs

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgbmv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a,
                                          lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, double alpha, const double *a,
                     std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgbmv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a,
                                          lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgbmv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a,
                                          lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgbmv_sycl(&queue, MAJOR, mkl_convert(trans), m, n, kl, ku, alpha, a,
                                          lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                    std::int64_t lda, const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sger_sycl(&queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                         dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                    double *a, std::int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dger_sycl(&queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                         dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgerc_sycl(&queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgerc_sycl(&queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgeru_sycl(&queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgeru_sycl(&queue, MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chbmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhbmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chemv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhemv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    std::int64_t lda, const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         lda, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    std::int64_t lda, const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                     std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, lda, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx,
                                          beta, y, incy, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx,
                                          beta, y, incy, dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpr_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpr_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
                     std::int64_t incy, std::complex<float> *a,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpr2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpr2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                     float beta, float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssbmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsbmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, k, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha, const float *a,
                     const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx,
                                          beta, y, incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                     const double *a, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspmv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, x, incx,
                                          beta, y, incy, dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha, const float *x,
                    std::int64_t incx, float *a,
                    const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspr_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a,
                    const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspr_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspr2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspr2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssymv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsymv_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, a, lda, x,
                                          incx, beta, y, incy, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha, const float *x,
                    std::int64_t incx, float *a, std::int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         lda, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a, std::int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, a,
                                         lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, uplo uplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr2_sycl(&queue, MAJOR, mkl_convert(uplo), n, alpha, x, incx, y,
                                          incy, a, lda, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stbmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtbmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctbmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztbmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stbsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtbsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctbsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     std::int64_t k, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztbsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const float *a, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stpmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const double *a, double *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtpmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctpmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztpmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const float *a, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stpsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const double *a, double *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtpsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctpsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztpsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const double *a, std::int64_t lda, double *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrmv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const double *a, std::int64_t lda, double *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrsv_sycl(&queue, MAJOR, mkl_convert(uplo), mkl_convert(trans),
                                          mkl_convert(diag), n, a, lda, x, incx, dependencies);
}
