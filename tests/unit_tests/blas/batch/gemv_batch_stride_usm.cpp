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

#include <CL/sycl.hpp>
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
int test(device *dev, oneapi::mkl::layout layout, int64_t incx, int64_t incy, int64_t batch_size) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    int64_t m, n;
    int64_t lda;
    oneapi::mkl::transpose transa;
    fp alpha, beta;
    int64_t i, tmp;

    batch_size = 15;
    m = 25;
    n = 30;
    lda = 42;
    alpha = rand_scalar<fp>();
    beta = rand_scalar<fp>();

    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        transa = (oneapi::mkl::transpose)(std::rand() % 2);
    }
    else {
        tmp = std::rand() % 3;
        if (tmp == 2)
            transa = oneapi::mkl::transpose::conjtrans;
        else
            transa = (oneapi::mkl::transpose)tmp;
    }

    int x_len = outer_dimension(transa, m, n);
    int y_len = inner_dimension(transa, m, n);

    int64_t stride_x, stride_y, stride_a;
    stride_x = x_len * std::abs(incx);
    stride_y = y_len * std::abs(incy);
    stride_a = lda * std::max(m, n);

    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> x(ua), y(ua), A(ua), y_ref(ua);

    x.resize(stride_x * batch_size);
    y.resize(stride_y * batch_size);
    A.resize(stride_a * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_vector(&x[stride_x * i], x_len, incx);
        rand_vector(&y[stride_y * i], y_len, incy);
        rand_matrix(&A[stride_a * i], layout, oneapi::mkl::transpose::nontrans, m, n, lda);
    }

    y_ref.resize(y.size());
    for (int i = 0; i < y.size(); i++)
        y_ref[i] = y[i];

    // Call reference GEMV_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int m_ref = (int)m;
    int n_ref = (int)n;
    int incx_ref = (int)incx;
    int incy_ref = (int)incy;
    int lda_ref = (int)lda;
    int batch_size_ref = (int)batch_size;

    for (i = 0; i < batch_size_ref; i++) {
        ::gemv(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa), (const int *)&m_ref,
               (const int *)&n_ref, (const fp_ref *)&alpha, (const fp_ref *)&A[stride_a * i],
               (const int *)&lda_ref, (const fp_ref *)&x[stride_x * i], (const int *)&incx_ref,
               (const fp_ref *)&beta, (fp_ref *)&y_ref[stride_y * i], (const int *)&incy_ref);
    }

    // Call DPC++ GEMV_BATCH_STRIDE.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::gemv_batch(
                    main_queue, transa, m, n, alpha, &A[0], lda, stride_a, &x[0], incx, stride_x,
                    beta, &y[0], incy, stride_y, batch_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::gemv_batch(
                    main_queue, transa, m, n, alpha, &A[0], lda, stride_a, &x[0], incx, stride_x,
                    beta, &y[0], incy, stride_y, batch_size, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::gemv_batch, transa,
                                   m, n, alpha, &A[0], lda, stride_a, &x[0], incx, stride_x, beta,
                                   &y[0], incy, stride_y, batch_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::gemv_batch, transa, m,
                                   n, alpha, &A[0], lda, stride_a, &x[0], incx, stride_x, beta,
                                   &y[0], incy, stride_y, batch_size, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GEMV_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GEMV_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = true;
    for (i = 0; i < batch_size; i++) {
        good = good && check_equal_vector(&y[i * stride_y], &y_ref[i * stride_y], y_len, incy,
                                          std::max<int>(m, n), std::cout);
    }
    return (int)good;
}

class GemvBatchStrideUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(GemvBatchStrideUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

TEST_P(GemvBatchStrideUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

TEST_P(GemvBatchStrideUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5);
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5);
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

TEST_P(GemvBatchStrideUsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5);
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5);
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

INSTANTIATE_TEST_SUITE_P(GemvBatchStrideUsmTestSuite, GemvBatchStrideUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
