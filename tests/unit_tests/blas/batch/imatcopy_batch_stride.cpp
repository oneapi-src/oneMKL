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
    // Prepare data.
    int64_t m, n;
    int64_t lda, ldb;
    oneapi::mkl::transpose trans;
    fp alpha;
    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(m, n);
    alpha = rand_scalar<fp>();
    trans = rand_trans<fp>();

    int64_t stride_a, stride_b, stride;
    switch (layout) {
        case oneapi::mkl::layout::col_major:
            stride_a = lda * n;
            stride_b = (trans == oneapi::mkl::transpose::nontrans) ? ldb * n : ldb * m;
            stride = std::max(stride_a, stride_b);
            break;
        case oneapi::mkl::layout::row_major:
            stride_a = lda * m;
            stride_b = (trans == oneapi::mkl::transpose::nontrans) ? ldb * m : ldb * n;
            stride = std::max(stride_a, stride_b);
            break;
        default: break;
    }

    vector<fp, allocator_helper<fp, 64>> AB(stride * batch_size), AB_ref(stride * batch_size);

    rand_matrix(AB.data(), oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride * batch_size, 1, stride * batch_size);
    copy_matrix(AB.data(), oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride * batch_size, 1, stride * batch_size, AB_ref.data());

    // Call reference IMATCOPY_BATCH_STRIDE.
    int m_ref = (int)m;
    int n_ref = (int)n;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    int batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        imatcopy_ref(layout, trans, m_ref, n_ref, alpha, AB_ref.data() + stride * i, lda_ref,
                     ldb_ref);
    }

    // Call DPC++ IMATCOPY_BATCH_STRIDE

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during IMATCOPY_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> AB_buffer(AB.data(), range<1>(AB.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                oneapi::mkl::blas::column_major::imatcopy_batch(
                    main_queue, trans, m, n, alpha, AB_buffer, lda, ldb, stride, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::imatcopy_batch(
                    main_queue, trans, m, n, alpha, AB_buffer, lda, ldb, stride, batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::imatcopy_batch,
                                        trans, m, n, alpha, AB_buffer, lda, ldb, stride,
                                        batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::imatcopy_batch,
                                        trans, m, n, alpha, AB_buffer, lda, ldb, stride,
                                        batch_size);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during IMATCOPY_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of IMATCOPY_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto AB_accessor = AB_buffer.get_host_access(read_only);
    bool good = check_equal_matrix(AB_accessor, AB_ref, oneapi::mkl::layout::col_major,
                                   stride * batch_size, 1, stride * batch_size, 10, std::cout);

    return (int)good;
}

class ImatcopyBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(ImatcopyBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchStrideTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchStrideTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(ImatcopyBatchStrideTestSuite, ImatcopyBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
