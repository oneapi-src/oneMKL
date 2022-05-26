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

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_scopy_batch_strided>(cgh, [=]() {
            int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_scopy(n, accessor_x.get_pointer() + i * stridex, incx,
                              accessor_y.get_pointer() + i * stridey, incy);
            }
        });
    });
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
                int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dcopy_batch_strided>(cgh, [=]() {
            int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_dcopy(n, accessor_x.get_pointer() + i * stridex, incx,
                              accessor_y.get_pointer() + i * stridey, incy);
            }
        });
    });
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<float>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_ccopy_batch_strided>(cgh, [=]() {
            int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_ccopy(n, accessor_x.get_pointer() + i * stridex, incx,
                              accessor_y.get_pointer() + i * stridey, incy);
            }
        });
    });
}

void copy_batch(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<double>, 1> &y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zcopy_batch_strided>(cgh, [=]() {
            int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_zcopy(n, accessor_x.get_pointer() + i * stridex, incx,
                              accessor_y.get_pointer() + i * stridey, incy);
            }
        });
    });
}

void axpy_batch(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<float, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_saxpy_batch_strided>(cgh, [=]() {
            ::cblas_saxpy_batch_strided(n, alpha, accessor_x.get_pointer(), incx, stridex,
                                        accessor_y.get_pointer(), incy, stridey, batch_size);
        });
    });
}

void axpy_batch(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<double, 1> &x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_daxpy_batch_strided>(cgh, [=]() {
            ::cblas_daxpy_batch_strided(n, alpha, accessor_x.get_pointer(), incx, stridex,
                                        accessor_y.get_pointer(), incy, stridey, batch_size);
        });
    });
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_caxpy_batch_strided>(cgh, [=]() {
            ::cblas_caxpy_batch_strided(n, (const void *)&alpha, accessor_x.get_pointer(), incx,
                                        stridex, accessor_y.get_pointer(), incy, stridey,
                                        batch_size);
        });
    });
}

void axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zaxpy_batch_strided>(cgh, [=]() {
            ::cblas_zaxpy_batch_strided(n, (const void *)&alpha, accessor_x.get_pointer(), incx,
                                        stridex, accessor_y.get_pointer(), incy, stridey,
                                        batch_size);
        });
    });
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<float, 1> &x, int64_t incx, int64_t stride_x, float beta,
                sycl::buffer<float, 1> &y, int64_t incy, int64_t stride_y, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_sgemv_batch_strided>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **x_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **y_array = (float **)::malloc(sizeof(float *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                y_array[i] = y_acc.get_pointer() + i * stride_y;
            }
            ::cblas_sgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const float **)a_array, (const MKL_INT *)&lda,
                                (const float **)x_array, (const MKL_INT *)&incx, &beta,
                                (float **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_dgemv_batch_strided>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **x_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **y_array = (double **)::malloc(sizeof(double *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                y_array[i] = y_acc.get_pointer() + i * stride_y;
            }
            ::cblas_dgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const double **)a_array, (const MKL_INT *)&lda,
                                (const double **)x_array, (const MKL_INT *)&incx, &beta,
                                (double **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
                int64_t stride_x, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_cgemv_batch_strided>(cgh, [=]() {
            MKL_Complex8 **a_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **x_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **y_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                y_array[i] = y_acc.get_pointer() + i * stride_y;
            }
            ::cblas_cgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, &beta,
                                (void **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
}

void gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1> &x,
                int64_t incx, int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_zgemv_batch_strided>(cgh, [=]() {
            MKL_Complex16 **a_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            MKL_Complex16 **x_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            MKL_Complex16 **y_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                y_array[i] = y_acc.get_pointer() + i * stride_y;
            }
            ::cblas_zgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, &beta,
                                (void **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<float, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_sdgmm_batch_strided>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **x_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **c_array = (float **)::malloc(sizeof(float *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_sdgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                (const float **)a_array, (const MKL_INT *)&lda,
                                (const float **)x_array, (const MKL_INT *)&incx, (float **)c_array,
                                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_ddgmm_batch_strided>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **x_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **c_array = (double **)::malloc(sizeof(double *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_ddgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                (const double **)a_array, (const MKL_INT *)&lda,
                                (const double **)x_array, (const MKL_INT *)&incx,
                                (double **)c_array, (const MKL_INT *)&ldc, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_cdgmm_batch_strided>(cgh, [=]() {
            MKL_Complex8 **a_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **x_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **c_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_cdgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, (void **)c_array,
                                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
}

void dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_zdgmm_batch_strided>(cgh, [=]() {
            MKL_Complex16 **a_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            MKL_Complex16 **x_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            MKL_Complex16 **c_array =
                (MKL_Complex16 **)::malloc(sizeof(MKL_Complex16 *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                x_array[i] = x_acc.get_pointer() + i * stride_x;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_zdgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,

                                (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, (void **)c_array,
                                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_sgemm_batch_strided>(cgh, [=]() {
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
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_sgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const float **)a_array, (const MKL_INT *)&lda,
                (const float **)b_array, (const MKL_INT *)&ldb, &beta, (float **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                double beta, sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_dgemm_batch_strided>(cgh, [=]() {
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
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_dgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const double **)a_array, (const MKL_INT *)&lda,
                (const double **)b_array, (const MKL_INT *)&ldb, &beta, (double **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_cgemm_batch_strided>(cgh, [=]() {
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
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_cgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                (const void **)b_array, (const MKL_INT *)&ldb, &beta, (void **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b,
                int64_t ldb, int64_t stride_b, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_zgemm_batch_strided>(cgh, [=]() {
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
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
                c_array[i] = c_acc.get_pointer() + i * stride_c;
            }
            ::cblas_zgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                (const void **)b_array, (const MKL_INT *)&ldb, &beta, (void **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
}

void gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1> &a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1> &b, int64_t ldb, int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half",
                "half is not supported by the device or the sycl compiler");
        }
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_hgemm_batch_strided>(cgh, [=]() {
            int64_t totalsize_a, totalsize_b, totalsize_c;
            int64_t size_a, size_b, size_c;
#ifdef COLUMN_MAJOR
            size_a = (transa == transpose::N) ? lda * k : lda * m;
            size_b = (transb == transpose::N) ? ldb * n : ldb * k;
            size_c = ldc * n;
#endif
#ifdef ROW_MAJOR
            size_a = (transa == transpose::N) ? lda * m : lda * k;
            size_b = (transb == transpose::N) ? ldb * k : ldb * n;
            size_c = ldc * m;
#endif
            totalsize_a = (batch_size - 1) * stride_a + size_a;
            totalsize_b = (batch_size - 1) * stride_b + size_b;
            totalsize_c = (batch_size - 1) * stride_c + size_c;

            float *f32_a = (float *)::malloc(sizeof(float) * totalsize_a);
            float *f32_b = (float *)::malloc(sizeof(float) * totalsize_b);
            float *f32_c = (float *)::malloc(sizeof(float) * totalsize_c);
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **b_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **c_array = (float **)::malloc(sizeof(float *) * batch_size);
            // copy A, B and C to float
            copy_mat(a_acc, MKL_COL_MAJOR, transpose::N, totalsize_a, 1, totalsize_a, 0.0f, f32_a);
            copy_mat(b_acc, MKL_COL_MAJOR, transpose::N, totalsize_b, 1, totalsize_b, 0.0f, f32_b);
            copy_mat(c_acc, MKL_COL_MAJOR, transpose::N, totalsize_c, 1, totalsize_c, 0.0f, f32_c);
            float alphaf = (float)alpha, betaf = (float)beta;
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = f32_a + i * stride_a;
                b_array[i] = f32_b + i * stride_b;
                c_array[i] = f32_c + i * stride_c;
            }
            ::cblas_sgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alphaf, (const float **)a_array, (const MKL_INT *)&lda,
                (const float **)b_array, (const MKL_INT *)&ldb, &betaf, (float **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);
            // copy C back to half
            sycl::half co = 0.0f;
            copy_mat(f32_c, MKL_COL_MAJOR, totalsize_c, 1, totalsize_c, offset::F, &co,
                     (sycl::half *)c_acc.get_pointer());
            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
            ::free(f32_a);
            ::free(f32_b);
            ::free(f32_c);
        });
    });
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_strsm_batch_strided>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **b_array = (float **)::malloc(sizeof(float *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
            }
            ::cblas_strsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const float **)a_array,
                                (const MKL_INT *)&lda, (float **)b_array, (const MKL_INT *)&ldb,
                                one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1> &a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1> &b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_dtrsm_batch_strided>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **b_array = (double **)::malloc(sizeof(double *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
            }

            ::cblas_dtrsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const double **)a_array,
                                (const MKL_INT *)&lda, (double **)b_array, (const MKL_INT *)&ldb,
                                one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;

        host_task<class mkl_kernel_ctrsm_batch_strided>(cgh, [=]() {
            MKL_Complex8 **a_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            MKL_Complex8 **b_array = (MKL_Complex8 **)::malloc(sizeof(MKL_Complex8 *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
            }
            ::cblas_ctrsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const void **)a_array,
                                (const MKL_INT *)&lda, (void **)b_array, (const MKL_INT *)&ldb, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

void trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_ztrsm_batch_strided>(cgh, [=]() {
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
                a_array[i] = a_acc.get_pointer() + i * stride_a;
                b_array[i] = b_acc.get_pointer() + i * stride_b;
            }
            ::cblas_ztrsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const void **)a_array,
                                (const MKL_INT *)&lda, (void **)b_array, (const MKL_INT *)&ldb, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
        });
    });
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                float beta, sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_ssyrk_batch_strided>(cgh, [=]() {
            ::cblas_ssyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, alpha, a_acc.get_pointer(),
                                        lda, stride_a, beta, c_acc.get_pointer(), ldc, stride_c,
                                        batch_size);
        });
    });
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dsyrk_batch_strided>(cgh, [=]() {
            ::cblas_dsyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, alpha, a_acc.get_pointer(),
                                        lda, stride_a, beta, c_acc.get_pointer(), ldc, stride_c,
                                        batch_size);
        });
    });
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                int64_t stride_a, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_csyrk_batch_strided>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, (const void *)&alpha_,
                                        a_acc.get_pointer(), lda, stride_a, (const void *)&beta_,
                                        c_acc.get_pointer(), ldc, stride_c, batch_size);
        });
    });
}

void syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                int64_t lda, int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1> &c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zsyrk_batch_strided>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, (const void *)&alpha_,
                                        a_acc.get_pointer(), lda, stride_a, (const void *)&beta_,
                                        c_acc.get_pointer(), ldc, stride_c, batch_size);
        });
    });
}
void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_somatcopy_batch>(cgh, [=]() {
            ::mkl_simatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha,
                                          a_acc.get_pointer(), lda, stride_a, b_acc.get_pointer(),
                                          ldb, stride_b, batch_size);
        });
    });
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_domatcopy_batch>(cgh, [=]() {
            ::mkl_dimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha,
                                          a_acc.get_pointer(), lda, stride_a, b_acc.get_pointer(),
                                          ldb, stride_b, batch_size);
        });
    });
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<float> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_comatcopy_batch>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::mkl_cimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          a_acc.get_pointer(), lda, stride_a, b_acc.get_pointer(),
                                          ldb, stride_b, batch_size);
        });
    });
}

void omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<double> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zomatcopy_batch>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::mkl_zimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          a_acc.get_pointer(), lda, stride_a, b_acc.get_pointer(),
                                          ldb, stride_b, batch_size);
        });
    });
}
void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto ab_acc = ab.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_simatcopy_batch>(cgh, [=]() {
            ::mkl_simatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha,
                                          ab_acc.get_pointer(), lda, ldb, stride, batch_size);
        });
    });
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1> &ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto ab_acc = ab.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_dimatcopy_batch>(cgh, [=]() {
            ::mkl_dimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha,
                                          ab_acc.get_pointer(), lda, ldb, stride, batch_size);
        });
    });
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<float> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto ab_acc = ab.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_cimatcopy_batch>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::mkl_cimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          ab_acc.get_pointer(), lda, ldb, stride, batch_size);
        });
    });
}

void imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    queue.submit([&](sycl::handler &cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<double> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        auto ab_acc = ab.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class mkl_kernel_zimatcopy_batch>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::mkl_zimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          ab_acc.get_pointer(), lda, ldb, stride, batch_size);
        });
    });
}

// USM APIs

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const float **x, int64_t *incx,
                           float **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_scopy_batch_group_usm>(cgh, [=]() {
            int64_t i, j, offset = 0;
            for (i = 0; i < group_count; i++) {
                for (j = 0; j < group_size[i]; j++) {
                    ::cblas_scopy(n[i], x[offset], incx[i], y[offset], incy[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const double **x, int64_t *incx,
                           double **y, int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_dcopy_batch_group_usm>(cgh, [=]() {
            int64_t i, j, offset = 0;
            for (i = 0; i < group_count; i++) {
                for (j = 0; j < group_size[i]; j++) {
                    ::cblas_dcopy(n[i], x[offset], incx[i], y[offset], incy[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<float> **x,
                           int64_t *incx, std::complex<float> **y, int64_t *incy,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_ccopy_batch_group_usm>(cgh, [=]() {
            int64_t i, j, offset = 0;
            for (i = 0; i < group_count; i++) {
                for (j = 0; j < group_size[i]; j++) {
                    ::cblas_ccopy(n[i], x[offset], incx[i], y[offset], incy[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

sycl::event copy_batch(sycl::queue &queue, int64_t *n, const std::complex<double> **x,
                           int64_t *incx, std::complex<double> **y, int64_t *incy,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zcopy_batch_group_usm>(cgh, [=]() {
            int64_t i, j, offset = 0;
            for (i = 0; i < group_count; i++) {
                for (j = 0; j < group_size[i]; j++) {
                    ::cblas_zcopy(n[i], x[offset], incx[i], y[offset], incy[i]);
                    offset++;
                }
            }
        });
    });
    return done;
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const float *x, int64_t incx,
                           std::int64_t stridex, float *y, int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_scopy_batch_strided_usm>(cgh, [=]() {
            std::int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_scopy(n, x + i * stridex, incx, y + i * stridey, incy);
            }
        });
    });
    return done;
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                           std::int64_t stridex, double *y, int64_t incy, std::int64_t stridey,
                           std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_dcopy_batch_strided_usm>(cgh, [=]() {
            std::int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_dcopy(n, x + i * stridex, incx, y + i * stridey, incy);
            }
        });
    });
    return done;
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<float> *x,
                           int64_t incx, std::int64_t stridex, std::complex<float> *y, int64_t incy,
                           std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_ccopy_batch_strided_usm>(cgh, [=]() {
            std::int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_ccopy(n, x + i * stridex, incx, y + i * stridey, incy);
            }
        });
    });
    return done;
}

sycl::event copy_batch(sycl::queue &queue, int64_t n, const std::complex<double> *x,
                           int64_t incx, std::int64_t stridex, std::complex<double> *y,
                           int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zcopy_batch_strided_usm>(cgh, [=]() {
            std::int64_t i;
            for (i = 0; i < batch_size; i++) {
                ::cblas_zcopy(n, x + i * stridex, incx, y + i * stridey, incy);
            }
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, float *alpha, const float **x,
                           int64_t *incx, float **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_saxpy_batch_group_usm>(cgh, [=]() {
            ::cblas_saxpy_batch((const MKL_INT *)n, (const float *)alpha, x, (const MKL_INT *)incx,
                                y, (const MKL_INT *)incy, group_count, (const MKL_INT *)group_size);
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, double *alpha, const double **x,
                           int64_t *incx, double **y, int64_t *incy, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_daxpy_batch_group_usm>(cgh, [=]() {
            ::cblas_daxpy_batch((const MKL_INT *)n, (const double *)alpha, x, (const MKL_INT *)incx,
                                y, (const MKL_INT *)incy, group_count, (const MKL_INT *)group_size);
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_caxpy_batch_group_usm>(cgh, [=]() {
            ::cblas_caxpy_batch((const MKL_INT *)n, (const void *)alpha, (const void **)x,
                                (const MKL_INT *)incx, (void **)y, (const MKL_INT *)incy,
                                group_count, (const MKL_INT *)group_size);
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **y,
                           int64_t *incy, int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zaxpy_batch_group_usm>(cgh, [=]() {
            ::cblas_zaxpy_batch((const MKL_INT *)n, (const void *)alpha, (const void **)x,
                                (const MKL_INT *)incx, (void **)y, (const MKL_INT *)incy,
                                group_count, (const MKL_INT *)group_size);
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, float alpha, const float *x,
                           int64_t incx, int64_t stridex, float *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_saxpy_batch_strided_usm>(cgh, [=]() {
            ::cblas_saxpy_batch_strided(n, alpha, x, incx, stridex, y, incy, stridey, batch_size);
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, double alpha, const double *x,
                           int64_t incx, int64_t stridex, double *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_daxpy_batch_strided_usm>(cgh, [=]() {
            ::cblas_daxpy_batch_strided(n, alpha, x, incx, stridex, y, incy, stridey, batch_size);
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *x, int64_t incx, int64_t stridex,
                           std::complex<float> *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_caxpy_batch_strided_usm>(cgh, [=]() {
            ::cblas_caxpy_batch_strided(n, (const void *)&alpha, x, incx, stridex, y, incy, stridey,
                                        batch_size);
        });
    });
    return done;
}

sycl::event axpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *x, int64_t incx, int64_t stridex,
                           std::complex<double> *y, int64_t incy, int64_t stridey,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zaxpy_batch_strided_usm>(cgh, [=]() {
            ::cblas_zaxpy_batch_strided(n, (const void *)&alpha, x, incx, stridex, y, incy, stridey,
                                        batch_size);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           float alpha, const float *a, int64_t lda, int64_t stride_a,
                           const float *x, int64_t incx, int64_t stride_x, float beta, float *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_sgemv_batch_strided_usm>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **x_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **y_array = (float **)::malloc(sizeof(float *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (float *)a + i * stride_a;
                x_array[i] = (float *)x + i * stride_x;
                y_array[i] = (float *)y + i * stride_y;
            }
            ::cblas_sgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const float **)a_array, (const MKL_INT *)&lda,
                                (const float **)x_array, (const MKL_INT *)&incx, &beta,
                                (float **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           double alpha, const double *a, int64_t lda, int64_t stride_a,
                           const double *x, int64_t incx, int64_t stride_x, double beta, double *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_dgemv_batch_strided_usm>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **x_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **y_array = (double **)::malloc(sizeof(double *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (double *)a + i * stride_a;
                x_array[i] = (double *)x + i * stride_x;
                y_array[i] = (double *)y + i * stride_y;
            }
            ::cblas_dgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const double **)a_array, (const MKL_INT *)&lda,
                                (const double **)x_array, (const MKL_INT *)&incx, &beta,
                                (double **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, const std::complex<float> *x, int64_t incx,
                           int64_t stride_x, std::complex<float> beta, std::complex<float> *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_cgemv_batch_strided_usm>(cgh, [=]() {
            std::complex<float> **a_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            std::complex<float> **x_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            std::complex<float> **y_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (std::complex<float> *)a + i * stride_a;
                x_array[i] = (std::complex<float> *)x + i * stride_x;
                y_array[i] = (std::complex<float> *)y + i * stride_y;
            }
            ::cblas_cgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, &beta,
                                (void **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose transa, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, const std::complex<double> *x, int64_t incx,
                           int64_t stride_x, std::complex<double> beta, std::complex<double> *y,
                           int64_t incy, int64_t stride_y, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        MKL_INT one = 1;
        host_task<class mkl_kernel_zgemv_batch_strided_usm>(cgh, [=]() {
            std::complex<double> **a_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            std::complex<double> **x_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            std::complex<double> **y_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (y_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(y_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (std::complex<double> *)a + i * stride_a;
                x_array[i] = (std::complex<double> *)x + i * stride_x;
                y_array[i] = (std::complex<double> *)y + i * stride_y;
            }
            ::cblas_zgemv_batch(CBLASMAJOR, &transa_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, &beta,
                                (void **)y_array, (const MKL_INT *)&incy, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(y_array);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           float *alpha, const float **a, int64_t *lda, const float **x,
                           int64_t *incx, float *beta, float **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_sgemv_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if (transa_ == NULL) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
            }
            ::cblas_sgemv_batch(CBLASMAJOR, transa_, (const MKL_INT *)m, (const MKL_INT *)n, alpha,
                                (const float **)a, (const MKL_INT *)lda, (const float **)x,
                                (const MKL_INT *)incx, beta, y, (const MKL_INT *)incy, group_count,
                                (const MKL_INT *)groupsize);
            ::free(transa_);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           double *alpha, const double **a, int64_t *lda, const double **x,
                           int64_t *incx, double *beta, double **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_dgemv_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if (transa_ == NULL) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
            }
            ::cblas_dgemv_batch(CBLASMAJOR, transa_, (const MKL_INT *)m, (const MKL_INT *)n, alpha,
                                (const double **)a, (const MKL_INT *)lda, (const double **)x,
                                (const MKL_INT *)incx, beta, y, (const MKL_INT *)incy, group_count,
                                (const MKL_INT *)groupsize);
            ::free(transa_);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> *beta,
                           std::complex<float> **y, int64_t *incy, int64_t group_count,
                           int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_cgemv_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if (transa_ == NULL) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
            }
            ::cblas_cgemv_batch(CBLASMAJOR, transa_, (const MKL_INT *)m, (const MKL_INT *)n, alpha,
                                (const void **)a, (const MKL_INT *)lda, (const void **)x,
                                (const MKL_INT *)incx, beta, (void **)y, (const MKL_INT *)incy,
                                group_count, (const MKL_INT *)groupsize);
            ::free(transa_);
        });
    });
    return done;
}

sycl::event gemv_batch(sycl::queue &queue, transpose *transa, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, const std::complex<double> **x, int64_t *incx,
                           std::complex<double> *beta, std::complex<double> **y, int64_t *incy,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zgemv_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if (transa_ == NULL) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
            }
            ::cblas_zgemv_batch(CBLASMAJOR, transa_, (const MKL_INT *)m, (const MKL_INT *)n, alpha,
                                (const void **)a, (const MKL_INT *)lda, (const void **)x,
                                (const MKL_INT *)incx, beta, (void **)y, (const MKL_INT *)incy,
                                group_count, (const MKL_INT *)groupsize);
            ::free(transa_);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const float *a, int64_t lda, int64_t stride_a, const float *x,
                           int64_t incx, int64_t stride_x, float *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_sdgmm_batch_strided_usm>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **x_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **c_array = (float **)::malloc(sizeof(float *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (float *)a + i * stride_a;
                x_array[i] = (float *)x + i * stride_x;
                c_array[i] = (float *)c + i * stride_c;
            }
            ::cblas_sdgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                (const float **)a_array, (const MKL_INT *)&lda,
                                (const float **)x_array, (const MKL_INT *)&incx, (float **)c_array,
                                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const double *a, int64_t lda, int64_t stride_a, const double *x,
                           int64_t incx, int64_t stride_x, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_ddgmm_batch_strided_usm>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **x_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **c_array = (double **)::malloc(sizeof(double *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (double *)a + i * stride_a;
                x_array[i] = (double *)x + i * stride_x;
                c_array[i] = (double *)c + i * stride_c;
            }
            ::cblas_ddgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                (const double **)a_array, (const MKL_INT *)&lda,
                                (const double **)x_array, (const MKL_INT *)&incx,
                                (double **)c_array, (const MKL_INT *)&ldc, one,
                                (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const std::complex<float> *a, int64_t lda, int64_t stride_a,
                           const std::complex<float> *x, int64_t incx, int64_t stride_x,
                           std::complex<float> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_cdgmm_batch_strided_usm>(cgh, [=]() {
            std::complex<float> **a_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            std::complex<float> **x_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            std::complex<float> **c_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (std::complex<float> *)a + i * stride_a;
                x_array[i] = (std::complex<float> *)x + i * stride_x;
                c_array[i] = (std::complex<float> *)c + i * stride_c;
            }
            ::cblas_cdgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, (void **)c_array,
                                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side left_right, int64_t m, int64_t n,
                           const std::complex<double> *a, int64_t lda, int64_t stride_a,
                           const std::complex<double> *x, int64_t incx, int64_t stride_x,
                           std::complex<double> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        MKL_INT one = 1;
        host_task<class mkl_kernel_zdgmm_batch_strided_usm>(cgh, [=]() {
            std::complex<double> **a_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            std::complex<double> **x_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            std::complex<double> **c_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            if ((a_array == NULL) || (x_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(x_array);
                ::free(c_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (std::complex<double> *)a + i * stride_a;
                x_array[i] = (std::complex<double> *)x + i * stride_x;
                c_array[i] = (std::complex<double> *)c + i * stride_c;
            }
            ::cblas_zdgmm_batch(CBLASMAJOR, &left_right_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                                (const void **)a_array, (const MKL_INT *)&lda,
                                (const void **)x_array, (const MKL_INT *)&incx, (void **)c_array,
                                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(x_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const float **a, int64_t *lda, const float **x, int64_t *incx, float **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_sdgmm_batch_group_usm>(cgh, [=]() {
            CBLAS_SIDE *left_right_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            if (left_right_ == NULL) {
                std::cout << "Error cannot allocate side arrays\n";
                ::free(left_right_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                left_right_[i] = cblas_convert(left_right[i]);
            }
            ::cblas_sdgmm_batch(CBLASMAJOR, left_right_, (const MKL_INT *)m, (const MKL_INT *)n,
                                (const float **)a, (const MKL_INT *)lda, (const float **)x,
                                (const MKL_INT *)incx, c, (const MKL_INT *)ldc, group_count,
                                (const MKL_INT *)groupsize);
            ::free(left_right_);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const double **a, int64_t *lda, const double **x, int64_t *incx,
                           double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_ddgmm_batch_group_usm>(cgh, [=]() {
            CBLAS_SIDE *left_right_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            if (left_right_ == NULL) {
                std::cout << "Error cannot allocate side arrays\n";
                ::free(left_right_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                left_right_[i] = cblas_convert(left_right[i]);
            }
            ::cblas_ddgmm_batch(CBLASMAJOR, left_right_, (const MKL_INT *)m, (const MKL_INT *)n,
                                (const double **)a, (const MKL_INT *)lda, (const double **)x,
                                (const MKL_INT *)incx, c, (const MKL_INT *)ldc, group_count,
                                (const MKL_INT *)groupsize);
            ::free(left_right_);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **x, int64_t *incx, std::complex<float> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_cdgmm_batch_group_usm>(cgh, [=]() {
            CBLAS_SIDE *left_right_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            if (left_right_ == NULL) {
                std::cout << "Error cannot allocate side arrays\n";
                ::free(left_right_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                left_right_[i] = cblas_convert(left_right[i]);
            }
            ::cblas_cdgmm_batch(CBLASMAJOR, left_right_, (const MKL_INT *)m, (const MKL_INT *)n,
                                (const void **)a, (const MKL_INT *)lda, (const void **)x,
                                (const MKL_INT *)incx, (void **)c, (const MKL_INT *)ldc,
                                group_count, (const MKL_INT *)groupsize);
            ::free(left_right_);
        });
    });
    return done;
}

sycl::event dgmm_batch(sycl::queue &queue, side *left_right, int64_t *m, int64_t *n,
                           const std::complex<double> **a, int64_t *lda,
                           const std::complex<double> **x, int64_t *incx, std::complex<double> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zdgmm_batch_group_usm>(cgh, [=]() {
            CBLAS_SIDE *left_right_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            if (left_right_ == NULL) {
                std::cout << "Error cannot allocate side arrays\n";
                ::free(left_right_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                left_right_[i] = cblas_convert(left_right[i]);
            }
            ::cblas_zdgmm_batch(CBLASMAJOR, left_right_, (const MKL_INT *)m, (const MKL_INT *)n,
                                (const void **)a, (const MKL_INT *)lda, (const void **)x,
                                (const MKL_INT *)incx, (void **)c, (const MKL_INT *)ldc,
                                group_count, (const MKL_INT *)groupsize);
            ::free(left_right_);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, float *alpha, const float **a, int64_t *lda,
                           const float **b, int64_t *ldb, float *beta, float **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_sgemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_TRANSPOSE *transb_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
            }
            ::cblas_sgemm_batch(CBLASMAJOR, transa_, transb_, (const MKL_INT *)m,
                                (const MKL_INT *)n, (const MKL_INT *)k, alpha, (const float **)a,
                                (const MKL_INT *)lda, (const float **)b, (const MKL_INT *)ldb, beta,
                                c, (const MKL_INT *)ldc, group_count, (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, double *alpha, const double **a, int64_t *lda,
                           const double **b, int64_t *ldb, double *beta, double **c, int64_t *ldc,
                           int64_t group_count, int64_t *group_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_dgemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_TRANSPOSE *transb_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
            }
            ::cblas_dgemm_batch(CBLASMAJOR, transa_, transb_, (const MKL_INT *)m,
                                (const MKL_INT *)n, (const MKL_INT *)k, alpha, (const double **)a,
                                (const MKL_INT *)lda, (const double **)b, (const MKL_INT *)ldb,
                                beta, c, (const MKL_INT *)ldc, group_count,
                                (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<float> *alpha,
                           const std::complex<float> **a, int64_t *lda,
                           const std::complex<float> **b, int64_t *ldb, std::complex<float> *beta,
                           std::complex<float> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_cgemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_TRANSPOSE *transb_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
            }
            ::cblas_cgemm_batch(CBLASMAJOR, transa_, transb_, (const MKL_INT *)m,
                                (const MKL_INT *)n, (const MKL_INT *)k, alpha, (const void **)a,
                                (const MKL_INT *)lda, (const void **)b, (const MKL_INT *)ldb, beta,
                                (void **)c, (const MKL_INT *)ldc, group_count,
                                (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, std::complex<double> *alpha,
                           const std::complex<double> **a, int64_t *lda,
                           const std::complex<double> **b, int64_t *ldb, std::complex<double> *beta,
                           std::complex<double> **c, int64_t *ldc, int64_t group_count,
                           int64_t *group_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zgemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_TRANSPOSE *transb_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL)) {
                std::cout << "Error cannot allocate trans arrays\n";
                ::free(transa_);
                ::free(transb_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
            }
            ::cblas_zgemm_batch(CBLASMAJOR, transa_, transb_, (const MKL_INT *)m,
                                (const MKL_INT *)n, (const MKL_INT *)k, alpha, (const void **)a,
                                (const MKL_INT *)lda, (const void **)b, (const MKL_INT *)ldb, beta,
                                (void **)c, (const MKL_INT *)ldc, group_count,
                                (const MKL_INT *)group_size);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose *transa, transpose *transb, int64_t *m,
                           int64_t *n, int64_t *k, sycl::half *alpha, const sycl::half **a,
                           int64_t *lda, const sycl::half **b, int64_t *ldb, sycl::half *beta,
                           sycl::half **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half",
                "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_hgemm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *transa_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_TRANSPOSE *transb_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            float *alphaf = (float *)::malloc(sizeof(float) * group_count);
            float *betaf = (float *)::malloc(sizeof(float) * group_count);
            if ((transa_ == NULL) || (transb_ == NULL) || (alphaf == NULL) || (betaf == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(transa_);
                ::free(transb_);
                ::free(alphaf);
                ::free(betaf);
                return;
            }
            int64_t totalbatch_size = 0;
            for (int64_t i = 0; i < group_count; i++) {
                transa_[i] = cblas_convert(transa[i]);
                transb_[i] = cblas_convert(transb[i]);
                alphaf[i] = (float)alpha[i];
                betaf[i] = (float)beta[i];
                totalbatch_size += groupsize[i];
            }
            float **a_array = (float **)::malloc(sizeof(float *) * totalbatch_size);
            float **b_array = (float **)::malloc(sizeof(float *) * totalbatch_size);
            float **c_array = (float **)::malloc(sizeof(float *) * totalbatch_size);
            if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                ::free(c_array);
                return;
            }
            int64_t sizea, sizeb, sizec, idx;
            sycl::half co = 0.0f;
            for (int64_t i = 0, idx = 0; i < group_count; i++) {
#ifdef COLUMN_MAJOR
                sizea = (transa[i] == transpose::N) ? lda[i] * k[i] : lda[i] * m[i];
                sizeb = (transb[i] == transpose::N) ? ldb[i] * n[i] : ldb[i] * k[i];
                sizec = ldc[i] * n[i];
#endif
#ifdef ROW_MAJOR
                sizea = (transa[i] == transpose::N) ? lda[i] * m[i] : lda[i] * k[i];
                sizeb = (transb[i] == transpose::N) ? ldb[i] * k[i] : ldb[i] * n[i];
                sizec = ldc[i] * m[i];
#endif
                for (int64_t j = 0; j < groupsize[i]; j++, idx++) {
                    a_array[idx] = (float *)::malloc(sizeof(float) * sizea);
                    b_array[idx] = (float *)::malloc(sizeof(float) * sizeb);
                    c_array[idx] = (float *)::malloc(sizeof(float) * sizec);
                    copy_mat(a[idx], MKLMAJOR, transa[i], m[i], k[i], lda[i], 0.0f, a_array[idx]);
                    copy_mat(b[idx], MKLMAJOR, transb[i], k[i], n[i], ldb[i], 0.0f, b_array[idx]);
                    copy_mat(c[idx], MKLMAJOR, transpose::N, m[i], n[i], ldc[i], 0.0f,
                             c_array[idx]);
                }
            }
            ::cblas_sgemm_batch(
                CBLASMAJOR, transa_, transb_, (const MKL_INT *)m, (const MKL_INT *)n,
                (const MKL_INT *)k, alphaf, (const float **)a_array, (const MKL_INT *)lda,
                (const float **)b_array, (const MKL_INT *)ldb, betaf, (float **)c_array,
                (const MKL_INT *)ldc, group_count, (const MKL_INT *)groupsize);
            for (int64_t i = 0, idx = 0; i < group_count; i++) {
                sizec = ldc[i] * n[i];
                for (int64_t j = 0; j < groupsize[i]; j++, idx++) {
                    copy_mat(c_array[idx], MKLMAJOR, m[i], n[i], ldc[i], offset::F, &co, c[idx]);
                    ::free(a_array[idx]);
                    ::free(b_array[idx]);
                    ::free(c_array[idx]);
                }
            }
            ::free(alphaf);
            ::free(betaf);
            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
            ::free(transa_);
            ::free(transb_);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                           int64_t stride_a, const float *b, int64_t ldb, int64_t stride_b,
                           float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_sgemm_batch_strided_usm>(cgh, [=]() {
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
                a_array[i] = (float *)a + i * stride_a;
                b_array[i] = (float *)b + i * stride_b;
                c_array[i] = (float *)c + i * stride_c;
            }
            ::cblas_sgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const float **)a_array, (const MKL_INT *)&lda,
                (const float **)b_array, (const MKL_INT *)&ldb, &beta, (float **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                           int64_t stride_a, const double *b, int64_t ldb, int64_t stride_b,
                           double beta, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_dgemm_batch_strided_usm>(cgh, [=]() {
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
                a_array[i] = (double *)a + i * stride_a;
                b_array[i] = (double *)b + i * stride_b;
                c_array[i] = (double *)c + i * stride_c;
            }
            ::cblas_dgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const double **)a_array, (const MKL_INT *)&lda,
                (const double **)b_array, (const MKL_INT *)&ldb, &beta, (double **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, int64_t stride_a,
                           const std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_cgemm_batch_strided_usm>(cgh, [=]() {
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
                a_array[i] = (std::complex<float> *)a + i * stride_a;
                b_array[i] = (std::complex<float> *)b + i * stride_b;
                c_array[i] = (std::complex<float> *)c + i * stride_c;
            }
            ::cblas_cgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                (const void **)b_array, (const MKL_INT *)&ldb, &beta, (void **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda, int64_t stride_a,
                           const std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                           int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_zgemm_batch_strided_usm>(cgh, [=]() {
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
                a_array[i] = (std::complex<double> *)a + i * stride_a;
                b_array[i] = (std::complex<double> *)b + i * stride_b;
                c_array[i] = (std::complex<double> *)c + i * stride_c;
            }
            ::cblas_zgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alpha, (const void **)a_array, (const MKL_INT *)&lda,
                (const void **)b_array, (const MKL_INT *)&ldb, &beta, (void **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
        });
    });
    return done;
}

sycl::event gemm_batch(sycl::queue &queue, transpose transa, transpose transb, int64_t m,
                           int64_t n, int64_t k, sycl::half alpha, const sycl::half *a, int64_t lda,
                           int64_t stride_a, const sycl::half *b, int64_t ldb, int64_t stride_b,
                           sycl::half beta, sycl::half *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half",
                "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        MKL_INT one = 1;
        host_task<class mkl_kernel_hgemm_batch_strided_usm>(cgh, [=]() {
            int64_t totalsize_a, totalsize_b, totalsize_c;
            int64_t size_a, size_b, size_c;
#ifdef COLUMN_MAJOR
            size_a = (transa == transpose::N) ? lda * k : lda * m;
            size_b = (transb == transpose::N) ? ldb * n : ldb * k;
            size_c = ldc * n;
#endif
#ifdef ROW_MAJOR
            size_a = (transa == transpose::N) ? lda * m : lda * k;
            size_b = (transb == transpose::N) ? ldb * k : ldb * n;
            size_c = ldc * m;
#endif
            totalsize_a = (batch_size - 1) * stride_a + size_a;
            totalsize_b = (batch_size - 1) * stride_b + size_b;
            totalsize_c = (batch_size - 1) * stride_c + size_c;

            // copy A, B and C to float
            float *f32_a = (float *)::malloc(sizeof(float) * totalsize_a);
            float *f32_b = (float *)::malloc(sizeof(float) * totalsize_b);
            float *f32_c = (float *)::malloc(sizeof(float) * totalsize_c);
            if ((f32_a == NULL) || (f32_b == NULL) || (f32_c == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(f32_a);
                ::free(f32_b);
                ::free(f32_c);
                return;
            }
            copy_mat(a, MKL_COL_MAJOR, transpose::N, totalsize_a, 1, totalsize_a, 0.0f, f32_a);
            copy_mat(b, MKL_COL_MAJOR, transpose::N, totalsize_b, 1, totalsize_b, 0.0f, f32_b);
            copy_mat(c, MKL_COL_MAJOR, transpose::N, totalsize_c, 1, totalsize_c, 0.0f, f32_c);

            float alphaf = (float)alpha, betaf = (float)beta;
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
                a_array[i] = (float *)f32_a + i * stride_a;
                b_array[i] = (float *)f32_b + i * stride_b;
                c_array[i] = (float *)f32_c + i * stride_c;
            }
            ::cblas_sgemm_batch(
                CBLASMAJOR, &transa_, &transb_, (const MKL_INT *)&m, (const MKL_INT *)&n,
                (const MKL_INT *)&k, &alphaf, (const float **)a_array, (const MKL_INT *)&lda,
                (const float **)b_array, (const MKL_INT *)&ldb, &betaf, (float **)c_array,
                (const MKL_INT *)&ldc, one, (const MKL_INT *)&batch_size);

            sycl::half co = 0.0f;
            copy_mat(f32_c, MKL_COL_MAJOR, totalsize_c, 1, totalsize_c, offset::F, &co, c);
            ::free(a_array);
            ::free(b_array);
            ::free(c_array);
            ::free(f32_a);
            ::free(f32_b);
            ::free(f32_c);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n, float alpha,
                           const float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_strsm_batch_strided_usm>(cgh, [=]() {
            float **a_array = (float **)::malloc(sizeof(float *) * batch_size);
            float **b_array = (float **)::malloc(sizeof(float *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (float *)a + i * stride_a;
                b_array[i] = (float *)b + i * stride_b;
            }
            ::cblas_strsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const float **)a_array,
                                (const MKL_INT *)&lda, (float **)b_array, (const MKL_INT *)&ldb,
                                one, (const MKL_INT *)&batch_size);
            ::free(a_array);
            ::free(b_array);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_dtrsm_batch_strided_usm>(cgh, [=]() {
            double **a_array = (double **)::malloc(sizeof(double *) * batch_size);
            double **b_array = (double **)::malloc(sizeof(double *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (double *)a + i * stride_a;
                b_array[i] = (double *)b + i * stride_b;
            }
            ::cblas_dtrsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const double **)a_array,
                                (const MKL_INT *)&lda, (double **)b_array, (const MKL_INT *)&ldb,
                                one, (const MKL_INT *)&batch_size);
            ::free(a_array);
            ::free(b_array);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_ctrsm_batch_strided_usm>(cgh, [=]() {
            std::complex<float> **a_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            std::complex<float> **b_array =
                (std::complex<float> **)::malloc(sizeof(std::complex<float> *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (std::complex<float> *)a + i * stride_a;
                b_array[i] = (std::complex<float> *)b + i * stride_b;
            }
            ::cblas_ctrsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const void **)a_array,
                                (const MKL_INT *)&lda, (void **)b_array, (const MKL_INT *)&ldb, one,
                                (const MKL_INT *)&batch_size);
            ::free(a_array);
            ::free(b_array);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_SIDE side_ = cblas_convert(left_right);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        CBLAS_DIAG diag_ = cblas_convert(unit_diag);
        MKL_INT one = 1;
        host_task<class mkl_kernel_ztrsm_batch_strided_usm>(cgh, [=]() {
            std::complex<double> **a_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            std::complex<double> **b_array =
                (std::complex<double> **)::malloc(sizeof(std::complex<double> *) * batch_size);
            if ((a_array == NULL) || (b_array == NULL)) {
                std::cout << "Error cannot allocate input arrays\n";
                ::free(a_array);
                ::free(b_array);
                return;
            }
            for (int64_t i = 0; i < batch_size; i++) {
                a_array[i] = (std::complex<double> *)a + i * stride_a;
                b_array[i] = (std::complex<double> *)b + i * stride_b;
            }
            ::cblas_ztrsm_batch(CBLASMAJOR, &side_, &uplo_, &trans_, &diag_, (const MKL_INT *)&m,
                                (const MKL_INT *)&n, &alpha, (const void **)a_array,
                                (const MKL_INT *)&lda, (void **)b_array, (const MKL_INT *)&ldb, one,
                                (const MKL_INT *)&batch_size);
            ::free(a_array);
            ::free(b_array);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, float *alpha,
                           const float **a, int64_t *lda, float **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        MKL_INT one = 1;
        host_task<class mkl_kernel_strsm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_SIDE *side_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            CBLAS_DIAG *diag_ = (CBLAS_DIAG *)::malloc(sizeof(CBLAS_DIAG) * group_count);
            if ((trans_ == NULL) || (side_ == NULL) || (uplo_ == NULL) || (diag_ == NULL)) {
                std::cout << "Error cannot allocate parameter arrays\n";
                ::free(trans_);
                ::free(side_);
                ::free(uplo_);
                ::free(diag_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = (CBLAS_TRANSPOSE)cblas_convert(trans[i]);
                side_[i] = (CBLAS_SIDE)cblas_convert(left_right[i]);
                uplo_[i] = (CBLAS_UPLO)cblas_convert(upper_lower[i]);
                diag_[i] = (CBLAS_DIAG)cblas_convert(unit_diag[i]);
            }
            ::cblas_strsm_batch(CBLASMAJOR, side_, uplo_, trans_, diag_, (const MKL_INT *)m,
                                (const MKL_INT *)n, alpha, (const float **)a, (const MKL_INT *)lda,
                                (float **)b, (const MKL_INT *)ldb, group_count,
                                (const MKL_INT *)groupsize);
            free(trans_);
            free(side_);
            free(uplo_);
            free(diag_);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n, double *alpha,
                           const double **a, int64_t *lda, double **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        MKL_INT one = 1;
        host_task<class mkl_kernel_dtrsm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_SIDE *side_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            CBLAS_DIAG *diag_ = (CBLAS_DIAG *)::malloc(sizeof(CBLAS_DIAG) * group_count);
            if ((trans_ == NULL) || (side_ == NULL) || (uplo_ == NULL) || (diag_ == NULL)) {
                std::cout << "Error cannot allocate parameter arrays\n";
                ::free(trans_);
                ::free(side_);
                ::free(uplo_);
                ::free(diag_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = cblas_convert(trans[i]);
                side_[i] = cblas_convert(left_right[i]);
                uplo_[i] = cblas_convert(upper_lower[i]);
                diag_[i] = cblas_convert(unit_diag[i]);
            }
            ::cblas_dtrsm_batch(CBLASMAJOR, side_, uplo_, trans_, diag_, (const MKL_INT *)m,
                                (const MKL_INT *)n, alpha, (const double **)a, (const MKL_INT *)lda,
                                (double **)b, (const MKL_INT *)ldb, group_count,
                                (const MKL_INT *)groupsize);
            free(trans_);
            free(side_);
            free(uplo_);
            free(diag_);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n,
                           std::complex<float> *alpha, const std::complex<float> **a, int64_t *lda,
                           std::complex<float> **b, int64_t *ldb, int64_t group_count,
                           int64_t *groupsize, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        MKL_INT one = 1;
        host_task<class mkl_kernel_ctrsm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_SIDE *side_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            CBLAS_DIAG *diag_ = (CBLAS_DIAG *)::malloc(sizeof(CBLAS_DIAG) * group_count);
            if ((trans_ == NULL) || (side_ == NULL) || (uplo_ == NULL) || (diag_ == NULL)) {
                std::cout << "Error cannot allocate parameter arrays\n";
                ::free(trans_);
                ::free(side_);
                ::free(uplo_);
                ::free(diag_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = cblas_convert(trans[i]);
                side_[i] = cblas_convert(left_right[i]);
                uplo_[i] = cblas_convert(upper_lower[i]);
                diag_[i] = cblas_convert(unit_diag[i]);
            }
            ::cblas_ctrsm_batch(CBLASMAJOR, side_, uplo_, trans_, diag_, (const MKL_INT *)m,
                                (const MKL_INT *)n, alpha, (const void **)a, (const MKL_INT *)lda,
                                (void **)b, (const MKL_INT *)ldb, group_count,
                                (const MKL_INT *)groupsize);
            free(trans_);
            free(side_);
            free(uplo_);
            free(diag_);
        });
    });
    return done;
}

sycl::event trsm_batch(sycl::queue &queue, side *left_right, uplo *upper_lower,
                           transpose *trans, diag *unit_diag, int64_t *m, int64_t *n,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        MKL_INT one = 1;
        host_task<class mkl_kernel_ztrsm_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_SIDE *side_ = (CBLAS_SIDE *)::malloc(sizeof(CBLAS_SIDE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            CBLAS_DIAG *diag_ = (CBLAS_DIAG *)::malloc(sizeof(CBLAS_DIAG) * group_count);
            if ((trans_ == NULL) || (side_ == NULL) || (uplo_ == NULL) || (diag_ == NULL)) {
                std::cout << "Error cannot allocate parameter arrays\n";
                ::free(trans_);
                ::free(side_);
                ::free(uplo_);
                ::free(diag_);
                return;
            }
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = cblas_convert(trans[i]);
                side_[i] = cblas_convert(left_right[i]);
                uplo_[i] = cblas_convert(upper_lower[i]);
                diag_[i] = cblas_convert(unit_diag[i]);
            }
            ::cblas_ztrsm_batch(CBLASMAJOR, side_, uplo_, trans_, diag_, (const MKL_INT *)m,
                                (const MKL_INT *)n, alpha, (const void **)a, (const MKL_INT *)lda,
                                (void **)b, (const MKL_INT *)ldb, group_count,
                                (const MKL_INT *)groupsize);
            free(trans_);
            free(side_);
            free(uplo_);
            free(diag_);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, float *alpha, const float **a, int64_t *lda, float *beta,
                           float **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_ssyrk_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = cblas_convert(trans[i]);
                uplo_[i] = cblas_convert(upper_lower[i]);
            }
            ::cblas_ssyrk_batch(CBLASMAJOR, uplo_, trans_, (const MKL_INT *)n, (const MKL_INT *)k,
                                alpha, (const float **)a, (const MKL_INT *)lda, beta, (float **)c,
                                (const MKL_INT *)ldc, group_count, (const MKL_INT *)groupsize);
            free(trans_);
            free(uplo_);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, double *alpha, const double **a, int64_t *lda, double *beta,
                           double **c, int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_dsyrk_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = cblas_convert(trans[i]);
                uplo_[i] = cblas_convert(upper_lower[i]);
            }
            ::cblas_dsyrk_batch(CBLASMAJOR, uplo_, trans_, (const MKL_INT *)n, (const MKL_INT *)k,
                                alpha, (const double **)a, (const MKL_INT *)lda, beta, (double **)c,
                                (const MKL_INT *)ldc, group_count, (const MKL_INT *)groupsize);
            free(trans_);
            free(uplo_);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, std::complex<float> *alpha, const std::complex<float> **a,
                           int64_t *lda, std::complex<float> *beta, std::complex<float> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_csyrk_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = cblas_convert(trans[i]);
                uplo_[i] = cblas_convert(upper_lower[i]);
            }
            ::cblas_csyrk_batch(CBLASMAJOR, uplo_, trans_, (const MKL_INT *)n, (const MKL_INT *)k,
                                (const void *)alpha, (const void **)a, (const MKL_INT *)lda,
                                (const void *)beta, (void **)c, (const MKL_INT *)ldc, group_count,
                                (const MKL_INT *)groupsize);
            free(trans_);
            free(uplo_);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo *upper_lower, transpose *trans, int64_t *n,
                           int64_t *k, std::complex<double> *alpha, const std::complex<double> **a,
                           int64_t *lda, std::complex<double> *beta, std::complex<double> **c,
                           int64_t *ldc, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class mkl_kernel_zsyrk_batch_group_usm>(cgh, [=]() {
            CBLAS_TRANSPOSE *trans_ =
                (CBLAS_TRANSPOSE *)::malloc(sizeof(CBLAS_TRANSPOSE) * group_count);
            CBLAS_UPLO *uplo_ = (CBLAS_UPLO *)::malloc(sizeof(CBLAS_UPLO) * group_count);
            for (int64_t i = 0; i < group_count; i++) {
                trans_[i] = cblas_convert(trans[i]);
                uplo_[i] = cblas_convert(upper_lower[i]);
            }
            ::cblas_zsyrk_batch(CBLASMAJOR, uplo_, trans_, (const MKL_INT *)n, (const MKL_INT *)k,
                                (const void *)alpha, (const void **)a, (const MKL_INT *)lda,
                                (const void *)beta, (void **)c, (const MKL_INT *)ldc, group_count,
                                (const MKL_INT *)groupsize);
            free(trans_);
            free(uplo_);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, float alpha, const float *a, int64_t lda, int64_t stride_a,
                           float beta, float *c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_ssyrk_batch_strided_usm>(cgh, [=]() {
            ::cblas_ssyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, alpha, a, lda, stride_a,
                                        beta, c, ldc, stride_c, batch_size);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, double alpha, const double *a, int64_t lda, int64_t stride_a,
                           double beta, double *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        host_task<class mkl_kernel_dsyrk_batch_strided_usm>(cgh, [=]() {
            ::cblas_dsyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, alpha, a, lda, stride_a,
                                        beta, c, ldc, stride_c, batch_size);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                           int64_t lda, int64_t stride_a, std::complex<float> beta,
                           std::complex<float> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        float alpha_real = alpha.real(), alpha_imag = alpha.imag();
        float beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_csyrk_batch_strided_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex8 beta_ = { beta_real, beta_imag };
            ::cblas_csyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, (const void *)&alpha_, a,
                                        lda, stride_a, (const void *)&beta_, c, ldc, stride_c,
                                        batch_size);
        });
    });
    return done;
}

sycl::event syrk_batch(sycl::queue &queue, uplo upper_lower, transpose trans, int64_t n,
                           int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                           int64_t lda, int64_t stride_a, std::complex<double> beta,
                           std::complex<double> *c, int64_t ldc, int64_t stride_c,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_UPLO uplo_ = cblas_convert(upper_lower);
        double alpha_real = alpha.real(), alpha_imag = alpha.imag();
        double beta_real = beta.real(), beta_imag = beta.imag();
        host_task<class mkl_kernel_zsyrk_batch_strided_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            MKL_Complex16 beta_ = { beta_real, beta_imag };
            ::cblas_zsyrk_batch_strided(CBLASMAJOR, uplo_, trans_, n, k, (const void *)&alpha_, a,
                                        lda, stride_a, (const void *)&beta_, c, ldc, stride_c,
                                        batch_size);
        });
    });
    return done;
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float *a, int64_t lda, int64_t stride_a, float *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_somatcopy_batch_usm>(cgh, [=]() {
            ::mkl_simatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha, a, lda, stride_a,
                                          b, ldb, stride_b, batch_size, dependencies);
        });
    });
    return done;
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, int64_t stride_a, double *b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_domatcopy_batch_usm>(cgh, [=]() {
            ::mkl_dimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha, a, lda, stride_a,
                                          b, ldb, stride_b, batch_size, dependencies);
        });
    });
    return done;
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           int64_t stride_a, std::complex<float> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<float> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_comatcopy_batch_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::mkl_cimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          a, lda, stride_a, b, ldb, stride_b, batch_size,
                                          dependencies);
        });
    });
    return done;
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           int64_t stride_a, std::complex<double> *b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<double> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_zomatcopy_batch_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::mkl_zimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          a, lda, stride_a, b, ldb, stride_b, batch_size,
                                          dependencies);
        });
    });
    return done;
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_simatcopy_batch_usm>(cgh, [=]() {
            ::mkl_simatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha, ab, lda, ldb,
                                          stride, batch_size, dependencies);
        });
    });
    return done;
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double *ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class mkl_kernel_dimatcopy_batch_usm>(cgh, [=]() {
            ::mkl_dimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, alpha, ab, lda, ldb,
                                          stride, batch_size, dependencies);
        });
    });
    return done;
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<float> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_cimatcopy_batch_usm>(cgh, [=]() {
            MKL_Complex8 alpha_ = { alpha_real, alpha_imag };
            ::mkl_cimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          ab, lda, ldb, stride, batch_size, dependencies);
        });
    });
    return done;
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double> *ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event> &dependencies) {
    auto done = queue.submit([&](sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        std::complex<double> alpha_real = alpha.real(), alpha_imag = alpha.imag();
        host_task<class mkl_kernel_zimatcopy_batch_usm>(cgh, [=]() {
            MKL_Complex16 alpha_ = { alpha_real, alpha_imag };
            ::mkl_zimatcopy_batch_strided(&queue, CBLASMAJOR, trans_, m, n, (const void *)&alpha_,
                                          ab, lda, ldb, stride, batch_size, dependencies);
        });
    });
    return done;
}
