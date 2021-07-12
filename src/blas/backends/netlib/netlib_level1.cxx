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

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_sasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_sasum((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_dasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dasum((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_scasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_scasum((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_dzasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dzasum((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_saxpy>(cgh, [=]() {
            ::cblas_saxpy((const int)n, (const float)alpha, accessor_x.get_pointer(),
                          (const int)incx, accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_daxpy>(cgh, [=]() {
            ::cblas_daxpy((const int)n, (const double)alpha, accessor_x.get_pointer(),
                          (const int)incx, accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_caxpy>(cgh, [=]() {
            ::cblas_caxpy((const int)n, (const void *)&alpha, accessor_x.get_pointer(),
                          (const int)incx, accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void axpy(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zaxpy>(cgh, [=]() {
            ::cblas_zaxpy((const int)n, (const void *)&alpha, accessor_x.get_pointer(),
                          (const int)incx, accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void axpby(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
           int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

void axpby(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
           int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

void axpby(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

void axpby(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_scopy>(cgh, [=]() {
            ::cblas_scopy((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dcopy>(cgh, [=]() {
            ::cblas_dcopy((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_ccopy>(cgh, [=]() {
            ::cblas_ccopy((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zcopy>(cgh, [=]() {
            ::cblas_zcopy((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_sdot>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_sdot((const int)n, accessor_x.get_pointer(), (const int)incx,
                             accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
         cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_ddot>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_ddot((const int)n, accessor_x.get_pointer(), (const int)incx,
                             accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_dsdot>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dsdot((const int)n, accessor_x.get_pointer(), (const int)incx,
                              accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void dotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cdotc>(cgh, [=]() {
            ::cblas_cdotc_sub((const int)n, accessor_x.get_pointer(), (const int)incx,
                              accessor_y.get_pointer(), (const int)incy,
                              accessor_result.get_pointer());
        });
    });
}

void dotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zdotc>(cgh, [=]() {
            ::cblas_zdotc_sub((const int)n, accessor_x.get_pointer(), (const int)incx,
                              accessor_y.get_pointer(), (const int)incy,
                              accessor_result.get_pointer());
        });
    });
}

void dotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cdotu>(cgh, [=]() {
            ::cblas_cdotu_sub((const int)n, accessor_x.get_pointer(), (const int)incx,
                              accessor_y.get_pointer(), (const int)incy,
                              accessor_result.get_pointer());
        });
    });
}

void dotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zdotu>(cgh, [=]() {
            ::cblas_zdotu_sub((const int)n, accessor_x.get_pointer(), (const int)incx,
                              accessor_y.get_pointer(), (const int)incy,
                              accessor_result.get_pointer());
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_isamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_isamin((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_idamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_idamin((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_icamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_icamin((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_izamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_izamin((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_isamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_isamax((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_idamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_idamax((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_icamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_icamax((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_izamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_izamax((int)n, accessor_x.get_pointer(), (int)incx);
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.template get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_snrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_snrm2((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_dnrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dnrm2((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_scnrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_scnrm2((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_dznrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dznrm2((const int)n, accessor_x.get_pointer(), (const int)std::abs(incx));
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, float c, float s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_srot>(cgh, [=]() {
            ::cblas_srot((const int)n, accessor_x.get_pointer(), (const int)incx,
                         accessor_y.get_pointer(), (const int)incy, (const float)c, (const float)s);
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
         cl::sycl::buffer<double, 1> &y, int64_t incy, double c, double s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_drot>(cgh, [=]() {
            ::cblas_drot((const int)n, accessor_x.get_pointer(), (const int)incx,
                         accessor_y.get_pointer(), (const int)incy, (const float)c, (const float)s);
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy, float c,
         float s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_csrot>(cgh, [=]() {
            ::cblas_csrot((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, (const float)c,
                          (const float)s);
        });
    });
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy, double c,
         double s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zdrot>(cgh, [=]() {
            ::cblas_zdrot((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, (const double)c,
                          (const double)s);
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_srotg>(cgh, [=]() {
            ::cblas_srotg(accessor_a.get_pointer(), accessor_b.get_pointer(),
                          accessor_c.get_pointer(), accessor_s.get_pointer());
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
          cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_drotg>(cgh, [=]() {
            ::cblas_drotg(accessor_a.get_pointer(), accessor_b.get_pointer(),
                          accessor_c.get_pointer(), accessor_s.get_pointer());
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_crotg>(cgh, [=]() {
            ::cblas_crotg(accessor_a.get_pointer(), accessor_b.get_pointer(),
                          accessor_c.get_pointer(), accessor_s.get_pointer());
        });
    });
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_a = a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zrotg>(cgh, [=]() {
            ::cblas_zrotg(accessor_a.get_pointer(), accessor_b.get_pointer(),
                          accessor_c.get_pointer(), accessor_s.get_pointer());
        });
    });
}

void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class netlib_srotm>(cgh, [=]() {
            ::cblas_srotm((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_param.get_pointer());
        });
    });
}

void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read>(cgh);
        host_task<class netlib_drotm>(cgh, [=]() {
            ::cblas_drotm((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy, accessor_param.get_pointer());
        });
    });
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
           cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_d1 = d1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_d2 = d2.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_x1 = x1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_srotmg>(cgh, [=]() {
            ::cblas_srotmg(accessor_d1.get_pointer(), accessor_d2.get_pointer(),
                           accessor_x1.get_pointer(), (float)y1, accessor_param.get_pointer());
        });
    });
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
           cl::sycl::buffer<double, 1> &x1, double y1, cl::sycl::buffer<double, 1> &param) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_d1 = d1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_d2 = d2.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_x1 = x1.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_drotmg>(cgh, [=]() {
            ::cblas_drotmg(accessor_d1.get_pointer(), accessor_d2.get_pointer(),
                           accessor_x1.get_pointer(), (double)y1, accessor_param.get_pointer());
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sscal>(cgh, [=]() {
            ::cblas_sscal((const int)n, (const float)alpha, accessor_x.get_pointer(),
                          (const int)std::abs(incx));
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dscal>(cgh, [=]() {
            ::cblas_dscal((const int)n, (const double)alpha, accessor_x.get_pointer(),
                          (const int)std::abs(incx));
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cscal>(cgh, [=]() {
            ::cblas_cscal((const int)n, (const void *)&alpha, accessor_x.get_pointer(),
                          (const int)std::abs(incx));
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_csscal>(cgh, [=]() {
            ::cblas_csscal((const int)n, (const float)alpha, accessor_x.get_pointer(),
                           (const int)std::abs(incx));
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zscal>(cgh, [=]() {
            ::cblas_zscal((const int)n, (const void *)&alpha, accessor_x.get_pointer(),
                          (const int)std::abs(incx));
        });
    });
}

void scal(cl::sycl::queue &queue, int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zdscal>(cgh, [=]() {
            ::cblas_zdscal((const int)n, (const double)alpha, accessor_x.get_pointer(),
                           (const int)std::abs(incx));
        });
    });
}

void sdsdot(cl::sycl::queue &queue, int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<cl::sycl::access::mode::write>(cgh);
        host_task<class netlib_sdsdot>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_sdsdot((const int)n, (const float)sb, accessor_x.get_pointer(),
                               (const int)incx, accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_sswap>(cgh, [=]() {
            ::cblas_sswap((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_dswap>(cgh, [=]() {
            ::cblas_dswap((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_cswap>(cgh, [=]() {
            ::cblas_cswap((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    queue.submit([&](cl::sycl::handler &cgh) {
        auto accessor_x = x.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<cl::sycl::access::mode::read_write>(cgh);
        host_task<class netlib_zswap>(cgh, [=]() {
            ::cblas_zswap((const int)n, accessor_x.get_pointer(), (const int)incx,
                          accessor_y.get_pointer(), (const int)incy);
        });
    });
}

// USM APIs

cl::sycl::event asum(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sasum_usm>(
            cgh, [=]() { result[0] = ::cblas_sasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event asum(cl::sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dasum_usm>(
            cgh, [=]() { result[0] = ::cblas_dasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event asum(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_scasum_usm>(
            cgh, [=]() { result[0] = ::cblas_scasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event asum(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x, int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dzasum_usm>(
            cgh, [=]() { result[0] = ::cblas_dzasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event axpy(cl::sycl::queue &queue, int64_t n, float alpha, const float *x, int64_t incx,
                     float *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_saxpy_usm>(cgh, [=]() {
            ::cblas_saxpy((const int)n, (const float)alpha, x, (const int)incx, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event axpy(cl::sycl::queue &queue, int64_t n, double alpha, const double *x, int64_t incx,
                     double *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_daxpy_usm>(cgh, [=]() {
            ::cblas_daxpy((const int)n, (const double)alpha, x, (const int)incx, y,
                          (const int)incy);
        });
    });
    return done;
}

cl::sycl::event axpy(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, int64_t incx, std::complex<float> *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_caxpy_usm>(cgh, [=]() {
            ::cblas_caxpy((const int)n, (const void *)&alpha, x, (const int)incx, y,
                          (const int)incy);
        });
    });
    return done;
}

cl::sycl::event axpy(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, int64_t incx, std::complex<double> *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zaxpy_usm>(cgh, [=]() {
            ::cblas_zaxpy((const int)n, (const void *)&alpha, x, (const int)incx, y,
                          (const int)incy);
        });
    });
    return done;
}

cl::sycl::event axpby(cl::sycl::queue &queue, int64_t n, float alpha, const float *x, int64_t incx,
                      float beta, float *y, int64_t incy,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

cl::sycl::event axpby(cl::sycl::queue &queue, int64_t n, double alpha, const double *x,
                      int64_t incx, double beta, double *y, int64_t incy,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

cl::sycl::event axpby(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
                      const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                      std::complex<float> *y, int64_t incy,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

cl::sycl::event axpby(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
                      const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                      std::complex<double> *y, int64_t incy,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

cl::sycl::event copy(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx, float *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_scopy_usm>(
            cgh, [=]() { ::cblas_scopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

cl::sycl::event copy(cl::sycl::queue &queue, int64_t n, const double *x, int64_t incx, double *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dcopy_usm>(
            cgh, [=]() { ::cblas_dcopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

cl::sycl::event copy(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                     std::complex<float> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ccopy_usm>(
            cgh, [=]() { ::cblas_ccopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

cl::sycl::event copy(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x, int64_t incx,
                     std::complex<double> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zcopy_usm>(
            cgh, [=]() { ::cblas_zcopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

cl::sycl::event dot(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx, const float *y,
                    int64_t incy, float *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sdot_usm>(cgh, [=]() {
            result[0] = ::cblas_sdot((const int)n, x, (const int)incx, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event dot(cl::sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                    const double *y, int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_ddot_usm>(cgh, [=]() {
            result[0] = ::cblas_ddot((const int)n, x, (const int)incx, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event dot(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx, const float *y,
                    int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dsdot_usm>(cgh, [=]() {
            result[0] = ::cblas_dsdot((const int)n, x, (const int)incx, y, (const int)incy);
        });
    });
    return done;
}

cl::sycl::event dotc(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                     const std::complex<float> *y, int64_t incy, std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cdotc_usm>(cgh, [=]() {
            ::cblas_cdotc_sub((const int)n, x, (const int)incx, y, (const int)incy, result);
        });
    });
    return done;
}

cl::sycl::event dotc(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x, int64_t incx,
                     const std::complex<double> *y, int64_t incy, std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zdotc_usm>(cgh, [=]() {
            ::cblas_zdotc_sub((const int)n, x, (const int)incx, y, (const int)incy, result);
        });
    });
    return done;
}

cl::sycl::event dotu(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                     const std::complex<float> *y, int64_t incy, std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cdotu_usm>(cgh, [=]() {
            ::cblas_cdotu_sub((const int)n, x, (const int)incx, y, (const int)incy, result);
        });
    });
    return done;
}

cl::sycl::event dotu(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x, int64_t incx,
                     const std::complex<double> *y, int64_t incy, std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zdotu_usm>(cgh, [=]() {
            ::cblas_zdotu_sub((const int)n, x, (const int)incx, y, (const int)incy, result);
        });
    });
    return done;
}

cl::sycl::event iamin(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx,
                      int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_isamin_usm>(
            cgh, [=]() { result[0] = ::cblas_isamin((int)n, x, (int)incx); });
    });
    return done;
}

cl::sycl::event iamin(cl::sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                      int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_idamin_usm>(
            cgh, [=]() { result[0] = ::cblas_idamin((const int)n, x, (const int)incx); });
    });
    return done;
}

cl::sycl::event iamin(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                      int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_icamin_usm>(
            cgh, [=]() { result[0] = ::cblas_icamin((int)n, x, (int)incx); });
    });
    return done;
}

cl::sycl::event iamin(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x,
                      int64_t incx, int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_izamin_usm>(
            cgh, [=]() { result[0] = ::cblas_izamin((int)n, x, (int)incx); });
    });
    return done;
}

cl::sycl::event iamax(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx,
                      int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_isamax_usm>(
            cgh, [=]() { result[0] = ::cblas_isamax((int)n, x, (int)incx); });
    });
    return done;
}

cl::sycl::event iamax(cl::sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                      int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_idamax_usm>(
            cgh, [=]() { result[0] = ::cblas_idamax((int)n, x, (int)incx); });
    });
    return done;
}

cl::sycl::event iamax(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                      int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_icamax_usm>(
            cgh, [=]() { result[0] = ::cblas_icamax((int)n, x, (int)incx); });
    });
    return done;
}

cl::sycl::event iamax(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x,
                      int64_t incx, int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_izamax_usm>(
            cgh, [=]() { result[0] = ::cblas_izamax((int)n, x, (int)incx); });
    });
    return done;
}

cl::sycl::event nrm2(cl::sycl::queue &queue, int64_t n, const float *x, int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_snrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_snrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event nrm2(cl::sycl::queue &queue, int64_t n, const double *x, int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dnrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_dnrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event nrm2(cl::sycl::queue &queue, int64_t n, const std::complex<float> *x, int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_scnrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_scnrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event nrm2(cl::sycl::queue &queue, int64_t n, const std::complex<double> *x, int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dznrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_dznrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

cl::sycl::event rot(cl::sycl::queue &queue, int64_t n, float *x, int64_t incx, float *y,
                    int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_srot_usm>(cgh, [=]() {
            ::cblas_srot((const int)n, x, (const int)incx, y, (const int)incy, (const float)c,
                         (const float)s);
        });
    });
    return done;
}

cl::sycl::event rot(cl::sycl::queue &queue, int64_t n, double *x, int64_t incx, double *y,
                    int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_drot_usm>(cgh, [=]() {
            ::cblas_drot((const int)n, x, (const int)incx, y, (const int)incy, (const float)c,
                         (const float)s);
        });
    });
    return done;
}

cl::sycl::event rot(cl::sycl::queue &queue, int64_t n, std::complex<float> *x, int64_t incx,
                    std::complex<float> *y, int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_csrot_usm>(cgh, [=]() {
            ::cblas_csrot((const int)n, x, (const int)incx, y, (const int)incy, (const float)c,
                          (const float)s);
        });
    });
    return done;
}

cl::sycl::event rot(cl::sycl::queue &queue, int64_t n, std::complex<double> *x, int64_t incx,
                    std::complex<double> *y, int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zdrot_usm>(cgh, [=]() {
            ::cblas_zdrot((const int)n, x, (const int)incx, y, (const int)incy, (const double)c,
                          (const double)s);
        });
    });
    return done;
}

cl::sycl::event rotg(cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_srotg_usm>(cgh, [=]() { ::cblas_srotg(a, b, c, s); });
    });
    return done;
}

cl::sycl::event rotg(cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_drotg_usm>(cgh, [=]() { ::cblas_drotg(a, b, c, s); });
    });
    return done;
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b,
                     float *c, std::complex<float> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_crotg_usm>(cgh, [=]() { ::cblas_crotg(a, b, c, s); });
    });
    return done;
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b,
                     double *c, std::complex<double> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zrotg_usm>(cgh, [=]() { ::cblas_zrotg(a, b, c, s); });
    });
    return done;
}

cl::sycl::event rotm(cl::sycl::queue &queue, int64_t n, float *x, int64_t incx, float *y,
                     int64_t incy, float *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_srotm_usm>(cgh, [=]() {
            ::cblas_srotm((const int)n, x, (const int)incx, y, (const int)incy, param);
        });
    });
    return done;
}

cl::sycl::event rotm(cl::sycl::queue &queue, int64_t n, double *x, int64_t incx, double *y,
                     int64_t incy, double *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_drotm_usm>(cgh, [=]() {
            ::cblas_drotm((const int)n, x, (const int)incx, y, (const int)incy, param);
        });
    });
    return done;
}

cl::sycl::event rotmg(cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                      float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_srotmg_usm>(cgh,
                                           [=]() { ::cblas_srotmg(d1, d2, x1, (float)y1, param); });
    });
    return done;
}

cl::sycl::event rotmg(cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1,
                      double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_drotmg_usm>(
            cgh, [=]() { ::cblas_drotmg(d1, d2, x1, (double)y1, param); });
    });
    return done;
}

cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, float alpha, float *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sscal_usm>(cgh, [=]() {
            ::cblas_sscal((const int)n, (const float)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, double alpha, double *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dscal_usm>(cgh, [=]() {
            ::cblas_dscal((const int)n, (const double)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
                     std::complex<float> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cscal_usm>(cgh, [=]() {
            ::cblas_cscal((const int)n, (const void *)&alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, float alpha, std::complex<float> *x,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_csscal_usm>(cgh, [=]() {
            ::cblas_csscal((const int)n, (const float)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
                     std::complex<double> *x, int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zscal_usm>(cgh, [=]() {
            ::cblas_zscal((const int)n, (const void *)&alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

cl::sycl::event scal(cl::sycl::queue &queue, int64_t n, double alpha, std::complex<double> *x,
                     int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zdscal_usm>(cgh, [=]() {
            ::cblas_zdscal((const int)n, (const double)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

cl::sycl::event sdsdot(cl::sycl::queue &queue, int64_t n, float sb, const float *x, int64_t incx,
                       const float *y, int64_t incy, float *result,
                       const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sdsdot_usm>(cgh, [=]() {
            result[0] = ::cblas_sdsdot((const int)n, (const float)sb, x, (const int)incx, y,
                                       (const int)incy);
        });
    });
    return done;
}

cl::sycl::event swap(cl::sycl::queue &queue, int64_t n, float *x, int64_t incx, float *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_sswap_usm>(
            cgh, [=]() { ::cblas_sswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

cl::sycl::event swap(cl::sycl::queue &queue, int64_t n, double *x, int64_t incx, double *y,
                     int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_dswap_usm>(
            cgh, [=]() { ::cblas_dswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

cl::sycl::event swap(cl::sycl::queue &queue, int64_t n, std::complex<float> *x, int64_t incx,
                     std::complex<float> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_cswap_usm>(
            cgh, [=]() { ::cblas_cswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

cl::sycl::event swap(cl::sycl::queue &queue, int64_t n, std::complex<double> *x, int64_t incx,
                     std::complex<double> *y, int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    auto done = queue.submit([&](cl::sycl::handler &cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class netlib_zswap_usm>(
            cgh, [=]() { ::cblas_zswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}
