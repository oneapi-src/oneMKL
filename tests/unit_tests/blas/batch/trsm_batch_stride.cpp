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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "allocator_helper.hpp"
#include "cblas.h"
#include "onemkl/detail/config.hpp"
#include "onemkl/onemkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp>
int test(const device &dev) {
    // Prepare data.
    int64_t m, n;
    int64_t lda, ldb;
    onemkl::transpose trans;
    onemkl::side left_right;
    onemkl::uplo upper_lower;
    onemkl::diag unit_diag;
    fp alpha;
    int64_t batch_size;
    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m          = 1 + std::rand() % 50;
    n          = 1 + std::rand() % 50;
    lda        = std::max(m, n);
    ldb        = std::max(n, m);
    alpha      = rand_scalar<fp>();

    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        trans = (onemkl::transpose)(std::rand() % 2);
    }
    else {
        tmp = std::rand() % 3;
        if (tmp == 2)
            trans = onemkl::transpose::conjtrans;
        else
            trans = (onemkl::transpose)tmp;
    }
    left_right  = (onemkl::side)(std::rand() % 2);
    upper_lower = (onemkl::uplo)(std::rand() % 2);
    unit_diag   = (onemkl::diag)(std::rand() % 2);

    int64_t stride_a, stride_b;
    int64_t total_size_b;

    stride_a     = (left_right == onemkl::side::left) ? lda * m : lda * n;
    stride_b     = ldb * n;
    total_size_b = batch_size * stride_b;

    vector<fp, allocator_helper<fp, 64>> A(batch_size * stride_a), B(total_size_b),
        B_ref(total_size_b);

    for (i = 0; i < batch_size; i++) {
        if (left_right == onemkl::side::left)
            rand_trsm_matrix(A.data() + stride_a * i, trans, m, m, lda);
        else
            rand_trsm_matrix(A.data() + stride_a * i, trans, n, n, lda);
        rand_matrix(B.data() + stride_b * i, onemkl::transpose::nontrans, m, n, ldb);
    }

    B_ref = B;

    // Call reference TRSM_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int m_ref, n_ref, lda_ref, ldb_ref, batch_size_ref;
    m_ref          = (int)m;
    n_ref          = (int)n;
    lda_ref        = (int)lda;
    ldb_ref        = (int)ldb;
    batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        ::trsm(convert_to_cblas_side(left_right), convert_to_cblas_uplo(upper_lower),
               convert_to_cblas_trans(trans), convert_to_cblas_diag(unit_diag), (const int *)&m_ref,
               (const int *)&n_ref, (const fp_ref *)&alpha,
               (const fp_ref *)(A.data() + stride_a * i), (const int *)&lda_ref,
               (fp_ref *)(B_ref.data() + stride_b * i), (const int *)&ldb_ref);
    }

    // Call DPC++ TRSM_BATCH_STRIDE.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during TRSM_BATCH_STRIDE:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));

    try {
#ifdef CALL_RT_API
        onemkl::blas::trsm_batch(main_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                                 A_buffer, lda, stride_a, B_buffer, ldb, stride_b, batch_size);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::trsm_batch,
                    (main_queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, A_buffer,
                     lda, stride_a, B_buffer, ldb, stride_b, batch_size));

#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during TRSM_BATCH_STRIDE:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const onemkl::backend_unsupported_exception &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of TRSM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto B_accessor = B_buffer.template get_access<access::mode::read>();
        good            = check_equal_trsm_matrix(B_accessor, B_ref, total_size_b, 1, total_size_b,
                                       10 * std::max(m, n), std::cout);
    }

    return (int)good;
}

class TrsmBatchStrideTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(TrsmBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(GetParam()));
}

TEST_P(TrsmBatchStrideTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(GetParam()));
}

TEST_P(TrsmBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam()));
}

TEST_P(TrsmBatchStrideTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(TrsmBatchStrideTestSuite, TrsmBatchStrideTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
