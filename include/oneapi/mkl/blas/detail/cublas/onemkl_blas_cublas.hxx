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

// Buffer APIs

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result);

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result);

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result);

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result);

void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

void axpy_batch(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
                std::int64_t incx, std::int64_t stridex, cl::sycl::buffer<float, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void axpy_batch(cl::sycl::queue &queue, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &x, std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<double, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                std::int64_t stridex, cl::sycl::buffer<std::complex<float>, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                std::int64_t stridex, cl::sycl::buffer<std::complex<double>, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void axpby(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
           std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void axpby(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void axpby(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void axpby(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
           std::int64_t incy);

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

void copy_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                std::int64_t incx, std::int64_t stridex, cl::sycl::buffer<float, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void copy_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                std::int64_t incx, std::int64_t stridex, cl::sycl::buffer<double, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void copy_batch(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
                std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size);

void copy_batch(cl::sycl::queue &queue, std::int64_t n,
                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                std::int64_t stridex, cl::sycl::buffer<std::complex<double>, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &result);

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result);

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result);

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result);

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result);

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result);

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result);

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result);

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result);

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result);

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result);

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result);

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result);

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result);

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
         float s);

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
         double c, double s);

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s);

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s);

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s);

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
          cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s);

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s);

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s);

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &param);

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &param);

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
           cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param);

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
           cl::sycl::buffer<double, 1> &x1, double y1, cl::sycl::buffer<double, 1> &param);

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx);

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx);

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
            cl::sycl::buffer<float, 1> &result);

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy);

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stridea,
                cl::sycl::buffer<float, 1> &x, std::int64_t incx, std::int64_t stridex, float beta,
                cl::sycl::buffer<float, 1> &y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size);

void gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stridea, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                std::int64_t stridex, double beta, cl::sycl::buffer<double, 1> &y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stridea, cl::sycl::buffer<std::complex<float>, 1> &x,
                std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size);

void gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stridea,
                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                std::int64_t stridex, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size);

void dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stridea,
                cl::sycl::buffer<float, 1> &x, std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size);

void dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stridea,
                cl::sycl::buffer<double, 1> &x, std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<double, 1> &c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size);

void dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::int64_t stridea,
                cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                std::int64_t stridex, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size);

void dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stridea, cl::sycl::buffer<std::complex<double>, 1> &x,
                std::int64_t incx, std::int64_t stridex,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stridec, std::int64_t batch_size);

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda);

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda);

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy);

void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);

void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda);

void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda);

void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy);

void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy);

void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a);

void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a);

void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a);

void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a);

void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a);

void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a);

void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a);

void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a);

void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy);

void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy);

void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda);

void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda);

void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda);

void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda);

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx);

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx);

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx);

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx);

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx);

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx);

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx);

void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx);

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx);

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx);

void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx);

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx);

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx);

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx);

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx);

void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx);

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, sycl::half alpha, cl::sycl::buffer<sycl::half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, sycl::half beta,
          cl::sycl::buffer<sycl::half, 1> &c, std::int64_t ldc);

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<sycl::half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<bfloat16, 1> &a,
          std::int64_t lda, cl::sycl::buffer<bfloat16, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc);

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           float beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           double beta, cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc);

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc);

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                std::int64_t stride_a, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, double beta, cl::sycl::buffer<double, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
           cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
           cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc);

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb);

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb);

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb);

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb);

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, float beta, cl::sycl::buffer<float, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                std::int64_t lda, std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, double beta,
                cl::sycl::buffer<double, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size);

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                std::int64_t n, std::int64_t k, sycl::half alpha,
                cl::sycl::buffer<sycl::half, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                sycl::half beta, cl::sycl::buffer<sycl::half, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size);

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size);

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size);

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
           std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc);

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc);

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc);

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               cl::sycl::buffer<int8_t, 1> &b, std::int64_t ldb, int8_t bo, float beta,
               cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               cl::sycl::buffer<uint8_t, 1> &a, std::int64_t lda, uint8_t ao,
               cl::sycl::buffer<int8_t, 1> &b, std::int64_t ldb, int8_t bo, float beta,
               cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
               cl::sycl::buffer<uint8_t, 1> &a, std::int64_t lda, uint8_t ao,
               cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);

// USM APIs

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                           std::int64_t incx, std::int64_t stridex, float *y, std::int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                           std::int64_t incx, std::int64_t stridex, double *y, std::int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                           const std::complex<float> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<float> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                           const std::complex<double> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<double> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpby(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                      std::int64_t incx, const float beta, float *y, std::int64_t incy,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpby(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                      std::int64_t incx, const double beta, double *y, std::int64_t incy,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpby(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                      const std::complex<float> *x, std::int64_t incx,
                      const std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event axpby(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                      const std::complex<double> *x, std::int64_t incx,
                      const std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t *n, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t *n, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t *n, const std::complex<float> **x,
                           std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t *n, const std::complex<double> **x,
                           std::int64_t *incx, std::complex<double> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t n, const float *x,
                           std::int64_t incx, std::int64_t stridex, float *y, std::int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t n, const double *x,
                           std::int64_t incx, std::int64_t stridex, double *y, std::int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<float> *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event copy_batch(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<double> *y,
                           std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, float *result,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                    const double *y, std::int64_t incy, double *result,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, double *result,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                    std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c, float s,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                    std::int64_t incx, std::complex<double> *y, std::int64_t incy, double c,
                    double s, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                    std::int64_t incy, float c, float s,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                    std::int64_t incy, double c, double s,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotg(cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotg(cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b,
                     float *c, std::complex<float> *s,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b,
                     double *c, std::complex<double> *s,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, float *param,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy, double *param,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotmg(cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                      float *param, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event rotmg(cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1,
                      double *param, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, double *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, const float *x,
                       std::int64_t incx, const float *y, std::int64_t incy, float *result,
                       const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, double alpha, const double *a,
                     std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           float alpha, const float *a, std::int64_t lda, std::int64_t stridea,
                           const float *x, std::int64_t incx, std::int64_t stridex, float beta,
                           float *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           double alpha, const double *a, std::int64_t lda, std::int64_t stridea,
                           const double *x, std::int64_t incx, std::int64_t stridex, double beta,
                           double *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stridea, const std::complex<float> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
                           std::complex<float> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stridea, const std::complex<double> *x,
                           std::int64_t incx, std::int64_t stridex, std::complex<double> beta,
                           std::complex<double> *y, std::int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *trans, std::int64_t *m,
                           std::int64_t *n, float *alpha, const float **a, std::int64_t *lda,
                           const float **x, std::int64_t *incx, float *beta, float **y,
                           std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *trans, std::int64_t *m,
                           std::int64_t *n, double *alpha, const double **a, std::int64_t *lda,
                           const double **x, std::int64_t *incx, double *beta, double **y,
                           std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *trans, std::int64_t *m,
                           std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **a, std::int64_t *lda,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> *beta, std::complex<float> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemv_batch(cl::sycl::queue &queue, transpose *trans, std::int64_t *m,
                           std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **a, std::int64_t *lda,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> *beta, std::complex<double> **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                           const float *a, std::int64_t lda, std::int64_t stridea, const float *x,
                           std::int64_t incx, std::int64_t stridex, float *c, std::int64_t ldc,
                           std::int64_t stridec, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                           const double *a, std::int64_t lda, std::int64_t stridea, const double *x,
                           std::int64_t incx, std::int64_t stridex, double *c, std::int64_t ldc,
                           std::int64_t stridec, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                           const std::complex<float> *a, std::int64_t lda, std::int64_t stridea,
                           const std::complex<float> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stridec,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side left_right, std::int64_t m, std::int64_t n,
                           const std::complex<double> *a, std::int64_t lda, std::int64_t stridea,
                           const std::complex<double> *x, std::int64_t incx, std::int64_t stridex,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stridec,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, std::int64_t *m,
                           std::int64_t *n, const float **a, std::int64_t *lda, const float **x,
                           std::int64_t *incx, float **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, std::int64_t *m,
                           std::int64_t *n, const double **a, std::int64_t *lda, const double **x,
                           std::int64_t *incx, double **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, std::int64_t *m,
                           std::int64_t *n, const std::complex<float> **a, std::int64_t *lda,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event dgmm_batch(cl::sycl::queue &queue, side *left_right, std::int64_t *m,
                           std::int64_t *n, const std::complex<double> **a, std::int64_t *lda,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                    std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                    double *a, std::int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                     const float *a, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                     const double *a, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                     float beta, float *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a, std::int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a, std::int64_t lda,
                    const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     std::int64_t lda, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, std::int64_t lda,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const float *a, std::int64_t lda, float *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const float *a, std::int64_t lda, float *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, std::int64_t k, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const float *a, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const double *a, double *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<float> *a, std::complex<float> *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<double> *a, std::complex<double> *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const float *a, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const double *a, double *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<float> *a, std::complex<float> *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<double> *a, std::complex<double> *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                     std::int64_t n, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *x, std::int64_t incx,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                     const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                     std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, sycl::half alpha, const sycl::half *a,
                     std::int64_t lda, const sycl::half *b, std::int64_t ldb, sycl::half beta,
                     sycl::half *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, float alpha, const sycl::half *a,
                     std::int64_t lda, const sycl::half *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, float alpha, const bfloat16 *a,
                     std::int64_t lda, const bfloat16 *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, float alpha, const std::complex<float> *a, std::int64_t lda,
                     float beta, std::complex<float> *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, double alpha, const std::complex<double> *a, std::int64_t lda,
                     double beta, std::complex<double> *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, float beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                      double beta, std::complex<double> *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *b,
                     std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, float alpha, const float *a, std::int64_t lda, float beta,
                     float *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, double alpha, const double *a, std::int64_t lda, double beta,
                     double *c, std::int64_t ldc,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans,
                           std::int64_t *n, std::int64_t *k, float *alpha, const float **a,
                           std::int64_t *lda, float *beta, float **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans,
                           std::int64_t *n, std::int64_t *k, double *alpha, const double **a,
                           std::int64_t *lda, double *beta, double **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans,
                           std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
                           const std::complex<float> **a, std::int64_t *lda,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo *upper_lower, transpose *trans,
                           std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
                           const std::complex<double> **a, std::int64_t *lda,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, float alpha, const float *a,
                           std::int64_t lda, std::int64_t stride_a, float beta, float *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, double alpha, const double *a,
                           std::int64_t lda, std::int64_t stride_a, double beta, double *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, std::complex<float> alpha,
                           const std::complex<float> *a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syrk_batch(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b,
                      std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, double alpha, const double *a, std::int64_t lda,
                      const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                      std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
                     std::int64_t lda, float *b, std::int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
                     std::int64_t lda, double *b, std::int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *b,
                     std::int64_t ldb, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                     std::int64_t ldb, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
                     std::int64_t lda, float *b, std::int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
                     std::int64_t lda, double *b, std::int64_t ldb,
                     const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *b,
                     std::int64_t ldb, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                     std::int64_t ldb, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           float alpha, const float *a, std::int64_t lda, std::int64_t stride_a,
                           float *b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           double alpha, const double *a, std::int64_t lda, std::int64_t stride_a,
                           double *b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, std::int64_t *m, std::int64_t *n,
                           float *alpha, const float **a, std::int64_t *lda, float **b,
                           std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, std::int64_t *m, std::int64_t *n,
                           double *alpha, const double **a, std::int64_t *lda, double **b,
                           std::int64_t *ldb, std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, std::int64_t *m, std::int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, std::complex<float> **b, std::int64_t *ldb,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event trsm_batch(cl::sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, std::int64_t *m, std::int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, std::complex<double> **b, std::int64_t *ldb,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha,
                           const float **a, std::int64_t *lda, const float **b, std::int64_t *ldb,
                           float *beta, float **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha,
                           const double **a, std::int64_t *lda, const double **b, std::int64_t *ldb,
                           double *beta, double **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, sycl::half *alpha,
                           const sycl::half **a, std::int64_t *lda, const sycl::half **b,
                           std::int64_t *ldb, sycl::half *beta, sycl::half **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a, const float *b,
                           std::int64_t ldb, std::int64_t stride_b, float beta, float *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                           const sycl::half *a, std::int64_t lda, std::int64_t stride_a,
                           const sycl::half *b, std::int64_t ldb, std::int64_t stride_b,
                           sycl::half beta, sycl::half *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                      const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                      std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                      std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc, const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                          float alpha, const std::int8_t *a, std::int64_t lda, std::int8_t ao,
                          const std::uint8_t *b, std::int64_t ldb, std::uint8_t bo, float beta,
                          std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                          float alpha, const std::int8_t *a, std::int64_t lda, std::int8_t ao,
                          const std::int8_t *b, std::int64_t ldb, std::int8_t bo, float beta,
                          std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                          float alpha, const std::uint8_t *a, std::int64_t lda, std::uint8_t ao,
                          const std::int8_t *b, std::int64_t ldb, std::int8_t bo, float beta,
                          std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies = {});

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                          float alpha, const std::uint8_t *a, std::int64_t lda, std::uint8_t ao,
                          const std::uint8_t *b, std::int64_t ldb, std::uint8_t bo, float beta,
                          std::int32_t *c, std::int64_t ldc, const std::int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies = {});
