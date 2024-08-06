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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "allocator_helper.hpp"
#include "cblas.h"
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device *> devices;

namespace {

template <typename fp>
int test(device *dev, oneapi::mkl::layout layout, int64_t incx, int64_t incy, fp alpha,
         int64_t batch_size) {
    // Prepare data.
    int64_t n, i;

    n = 1357;

    int64_t stride_x, stride_y;
    stride_x = n * std::abs(incx);
    stride_y = n * std::abs(incy);

    vector<fp, allocator_helper<fp, 64>> x(stride_x * batch_size);
    vector<fp, allocator_helper<fp, 64>> y(stride_y * batch_size), y_ref(stride_y * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_vector(x.data() + stride_x * i, n, incx);
        rand_vector(y.data() + stride_y * i, n, incy);
    }

    y_ref = y;

    // Call reference AXPY_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int n_ref = (int)n;
    int incx_ref = (int)incx;
    int incy_ref = (int)incy;
    int batch_size_ref = (int)batch_size;

    for (i = 0; i < batch_size_ref; i++) {
        ::axpy(&n_ref, (fp_ref *)&alpha, (fp_ref *)x.data() + i * stride_x, &incx_ref,
               (fp_ref *)y_ref.data() + i * stride_y, &incy_ref);
    }

    // Call DPC++ AXPY_BATCH_STRIDE.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during AXPY_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer(x.data(), range<1>(x.size()));
    buffer<fp, 1> y_buffer(y.data(), range<1>(y.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                oneapi::mkl::blas::column_major::axpy_batch(main_queue, n, alpha, x_buffer, incx,
                                                            stride_x, y_buffer, incy, stride_y,
                                                            batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::axpy_batch(main_queue, n, alpha, x_buffer, incx,
                                                         stride_x, y_buffer, incy, stride_y,
                                                         batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::axpy_batch, n,
                                        alpha, x_buffer, incx, stride_x, y_buffer, incy, stride_y,
                                        batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::axpy_batch, n,
                                        alpha, x_buffer, incx, stride_x, y_buffer, incy, stride_y,
                                        batch_size);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during AXPY_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of AXPY_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto y_accessor = y_buffer.get_host_access(read_only);
    bool good = true;
    for (i = 0; i < batch_size; i++) {
        good = good && check_equal_vector(y_accessor.get_pointer() + i * stride_y,
                                          y_ref.data() + i * stride_y, n, incy, n, std::cout);
    }
    return (int)good;
}

class AxpyBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(AxpyBatchStrideTests, RealSinglePrecision) {
    float alpha = 2.0;
    EXPECT_TRUEORSKIP(
        test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, alpha, 15));
    EXPECT_TRUEORSKIP(
        test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, alpha, 15));
    EXPECT_TRUEORSKIP(
        test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, alpha, 15));
}

TEST_P(AxpyBatchStrideTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    double alpha = 2.0;
    EXPECT_TRUEORSKIP(
        test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, alpha, 15));
    EXPECT_TRUEORSKIP(
        test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, alpha, 15));
    EXPECT_TRUEORSKIP(
        test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, alpha, 15));
}

TEST_P(AxpyBatchStrideTests, ComplexSinglePrecision) {
    std::complex<float> alpha = std::complex<float>(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2,
                                                3, alpha, 15));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                -2, -3, alpha, 15));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1,
                                                1, alpha, 15));
}

TEST_P(AxpyBatchStrideTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    std::complex<double> alpha = std::complex<double>(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 2, 3, alpha, 15));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 -2, -3, alpha, 15));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 1, 1, alpha, 15));
}

INSTANTIATE_TEST_SUITE_P(AxpyBatchStrideTestSuite, AxpyBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
