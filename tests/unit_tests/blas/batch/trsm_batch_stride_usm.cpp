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
int test(device *dev, oneapi::mkl::layout layout) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during TRSM_BATCH_STRIDE:\n"
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
    int64_t lda, ldb;
    oneapi::mkl::transpose trans;
    oneapi::mkl::side left_right;
    oneapi::mkl::uplo upper_lower;
    oneapi::mkl::diag unit_nonunit;
    fp alpha;
    int64_t batch_size;
    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(n, m);
    alpha = rand_scalar<fp>();

    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        trans = (oneapi::mkl::transpose)(std::rand() % 2);
    }
    else {
        tmp = std::rand() % 3;
        if (tmp == 2)
            trans = oneapi::mkl::transpose::conjtrans;
        else
            trans = (oneapi::mkl::transpose)tmp;
    }
    left_right = (oneapi::mkl::side)(std::rand() % 2);
    upper_lower = (oneapi::mkl::uplo)(std::rand() % 2);
    unit_nonunit = (oneapi::mkl::diag)(std::rand() % 2);

    int64_t stride_a, stride_b;
    int64_t total_size_b;

    stride_a = (left_right == oneapi::mkl::side::left) ? lda * m : lda * n;
    stride_b = (layout == oneapi::mkl::layout::column_major) ? ldb * n : ldb * m;

    total_size_b = batch_size * stride_b;

    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> A(ua), B(ua), B_ref(ua);

    A.resize(stride_a * batch_size);
    B.resize(total_size_b);
    B_ref.resize(total_size_b);

    for (i = 0; i < batch_size; i++) {
        if (left_right == oneapi::mkl::side::left)
            rand_trsm_matrix(&A[stride_a * i], layout, trans, m, m, lda);
        else
            rand_trsm_matrix(&A[stride_a * i], layout, trans, n, n, lda);
        rand_matrix(&B[stride_b * i], layout, oneapi::mkl::transpose::nontrans, m, n, ldb);
    }

    copy_matrix(B, oneapi::mkl::layout::column_major, oneapi::mkl::transpose::nontrans,
                total_size_b, 1, total_size_b, B_ref);

    // Call reference TRSM_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int m_ref, n_ref, lda_ref, ldb_ref, batch_size_ref;
    m_ref = (int)m;
    n_ref = (int)n;
    lda_ref = (int)lda;
    ldb_ref = (int)ldb;
    batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        ::trsm(convert_to_cblas_layout(layout), convert_to_cblas_side(left_right),
               convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
               convert_to_cblas_diag(unit_nonunit), (const int *)&m_ref, (const int *)&n_ref,
               (const fp_ref *)&alpha, (const fp_ref *)(A.data() + stride_a * i),
               (const int *)&lda_ref, (fp_ref *)(B_ref.data() + stride_b * i),
               (const int *)&ldb_ref);
    }

    // Call DPC++ TRSM_BATCH_STRIDE.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::trsm_batch(
                    main_queue, left_right, upper_lower, trans, unit_nonunit, m, n, alpha, &A[0],
                    lda, stride_a, &B[0], ldb, stride_b, batch_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::trsm_batch(
                    main_queue, left_right, upper_lower, trans, unit_nonunit, m, n, alpha, &A[0],
                    lda, stride_a, &B[0], ldb, stride_b, batch_size, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::trsm_batch,
                                   left_right, upper_lower, trans, unit_nonunit, m, n, alpha, &A[0],
                                   lda, stride_a, &B[0], ldb, stride_b, batch_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::trsm_batch, left_right,
                                   upper_lower, trans, unit_nonunit, m, n, alpha, &A[0], lda,
                                   stride_a, &B[0], ldb, stride_b, batch_size, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during TRSM_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of TRSM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_trsm_matrix(B, B_ref, oneapi::mkl::layout::column_major, total_size_b,
                                        1, total_size_b, 10 * std::max(m, n), std::cout);

    return (int)good;
}

class TrsmBatchStrideUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(TrsmBatchStrideUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(TrsmBatchStrideUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(TrsmBatchStrideUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

TEST_P(TrsmBatchStrideUsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(TrsmBatchStrideUsmTestSuite, TrsmBatchStrideUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
