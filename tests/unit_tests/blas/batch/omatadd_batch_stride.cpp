/*******************************************************************************
* Copyright 2022 Intel Corporation
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
int test(device *dev, oneapi::mkl::layout layout, int64_t batch_size) {
    // smoke test to make sure routine runs; there is no netlib reference
    // routine to test against

    // Prepare data.
    int64_t m, n;
    int64_t lda, ldb, ldc;
    oneapi::mkl::transpose transa, transb;
    fp alpha, beta;
    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(m, n);
    ldc = std::max(m, n);
    alpha = rand_scalar<fp>();
    beta = rand_scalar<fp>();

    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        transa = (oneapi::mkl::transpose)(std::rand() % 2);
        transb = (oneapi::mkl::transpose)(std::rand() % 2);
    }
    else {
        tmp = std::rand() % 3;
        if (tmp == 2)
            transa = oneapi::mkl::transpose::conjtrans;
        else
            transa = (oneapi::mkl::transpose)tmp;
        tmp = std::rand() % 3;
        if (tmp == 2)
            transb = oneapi::mkl::transpose::conjtrans;
        else
            transb = (oneapi::mkl::transpose)tmp;
    }

    int64_t stride_a, stride_b, stride_c;

    switch (layout) {
        case oneapi::mkl::layout::column_major:
            stride_a = (transa == oneapi::mkl::transpose::nontrans) ? lda * m : lda * n;
            stride_b = (transb == oneapi::mkl::transpose::nontrans) ? ldb * m : ldb * n;
            stride_c = ldc * m;
            break;
        case oneapi::mkl::layout::row_major:
            stride_a = (transa == oneapi::mkl::transpose::nontrans) ? lda * n : lda * m;
            stride_b = (transb == oneapi::mkl::transpose::nontrans) ? ldb * n : ldb * m;
            stride_c = lda * n;
            break;
        default: break;
    }

    vector<fp, allocator_helper<fp, 64>> A(stride_a * batch_size), B(stride_b * batch_size), C(stride_c * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_matrix(A.data() + stride_a * i, layout, transa, m, n, lda);
        rand_matrix(B.data() + stride_b * i, layout, transb, m, n, ldb);
        rand_matrix(C.data() + stride_c * i, layout, oneapi::mkl::transpose::nontrans, m, n, ldc);
    }

    // Call DPC++ OMATADD_BATCH_STRIDE

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during OMATADD_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::omatadd_batch(
                    main_queue, transa, transb, m, n, alpha, A_buffer, lda, stride_a,
                    beta, B_buffer, ldb, stride_b,
                    C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::omatadd_batch(
                    main_queue, transa, transb, m, n, alpha, A_buffer, lda, stride_a,
                    beta, B_buffer, ldb, stride_b,
                    C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::omatadd_batch,
                                   transa, transb, m, n, alpha, A_buffer, lda, stride_a,
                                   beta, B_buffer, ldb, stride_b,
                                   C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::omatadd_batch,
                                   transa, transb, m, n, alpha, A_buffer, lda, stride_a,
                                   beta, B_buffer, ldb, stride_b,
                                   C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during OMATADD_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of OMATADD_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    return 1;
}

class OmataddBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(OmataddBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmataddBatchStrideTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmataddBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmataddBatchStrideTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(OmataddBatchStrideTestSuite, OmataddBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
