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
bool test(const device &dev, int64_t group_count) {
    // Prepare data.
    int64_t *m   = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *n   = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *lda = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *ldb = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    onemkl::transpose *trans =
        (onemkl::transpose *)onemkl::aligned_alloc(64, sizeof(onemkl::transpose) * group_count);
    onemkl::side *left_right =
        (onemkl::side *)onemkl::aligned_alloc(64, sizeof(onemkl::side) * group_count);
    onemkl::uplo *upper_lower =
        (onemkl::uplo *)onemkl::aligned_alloc(64, sizeof(onemkl::uplo) * group_count);
    onemkl::diag *unit_diag =
        (onemkl::diag *)onemkl::aligned_alloc(64, sizeof(onemkl::diag) * group_count);
    fp *alpha           = (fp *)onemkl::aligned_alloc(64, sizeof(fp) * group_count);
    int64_t *group_size = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);

    if ((m == NULL) || (n == NULL) || (lda == NULL) || (ldb == NULL) || (trans == NULL) ||
        (left_right == NULL) || (upper_lower == NULL) || (unit_diag == NULL) || (alpha == NULL) ||
        (group_size == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        onemkl::aligned_free(m);
        onemkl::aligned_free(n);
        onemkl::aligned_free(lda);
        onemkl::aligned_free(ldb);
        onemkl::aligned_free(trans);
        onemkl::aligned_free(left_right);
        onemkl::aligned_free(upper_lower);
        onemkl::aligned_free(unit_diag);
        onemkl::aligned_free(alpha);
        onemkl::aligned_free(group_size);
        return false;
    }

    int64_t i, tmp;
    int64_t j, idx = 0, max = 0;
    int64_t total_size_a = 0, total_size_b = 0, total_batch_count = 0;
    int64_t size_a = 0, size_b = 0;
    int64_t off_a = 0, off_b = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        m[i]          = 1 + std::rand() % 50;
        n[i]          = 1 + std::rand() % 50;
        lda[i]        = std::max(m[i], n[i]);
        ldb[i]        = std::max(n[i], m[i]);
        alpha[i]      = rand_scalar<fp>();
        if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
            trans[i] = (onemkl::transpose)(std::rand() % 2);
        }
        else {
            tmp = std::rand() % 3;
            if (tmp == 2)
                trans[i] = onemkl::transpose::conjtrans;
            else
                trans[i] = (onemkl::transpose)tmp;
        }
        left_right[i]  = (onemkl::side)(std::rand() % 2);
        upper_lower[i] = (onemkl::uplo)(std::rand() % 2);
        unit_diag[i]   = (onemkl::diag)(std::rand() % 2);
    }

    for (i = 0; i < group_count; i++) {
        total_size_a +=
            lda[i] * group_size[i] * ((left_right[i] == onemkl::side::left) ? m[i] : n[i]);
        total_size_b += ldb[i] * group_size[i] * n[i];
        total_batch_count += group_size[i];
    }

    fp **a_array     = (fp **)onemkl::aligned_alloc(64, sizeof(fp *) * total_batch_count);
    fp **b_array     = (fp **)onemkl::aligned_alloc(64, sizeof(fp *) * total_batch_count);
    fp **b_ref_array = (fp **)onemkl::aligned_alloc(64, sizeof(fp *) * total_batch_count);

    if ((a_array == NULL) || (b_array == NULL) || (b_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        onemkl::aligned_free(a_array);
        onemkl::aligned_free(b_array);
        onemkl::aligned_free(b_ref_array);
        return false;
    }

    vector<fp, allocator_helper<fp, 64>> A(total_size_a), B(total_size_b), B_ref(total_size_b);

    for (i = 0; i < group_count; i++) {
        max    = std::max(max, m[i]);
        max    = std::max(max, n[i]);
        size_a = (left_right[i] == onemkl::side::left) ? m[i] * lda[i] : n[i] * lda[i];
        size_b = ldb[i] * n[i];
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx]     = A.data() + off_a;
            b_array[idx]     = B.data() + off_b;
            b_ref_array[idx] = B_ref.data() + off_b;
            if (left_right[i] == onemkl::side::left)
                rand_trsm_matrix(a_array[idx], trans[i], m[i], m[i], lda[i]);
            else
                rand_trsm_matrix(a_array[idx], trans[i], n[i], n[i], lda[i]);
            rand_matrix(b_array[idx], onemkl::transpose::nontrans, m[i], n[i], ldb[i]);
            off_a += size_a;
            off_b += size_b;
            idx++;
        }
    }

    B_ref = B;

    // Call reference TRSM_BATCH.
    using fp_ref = typename ref_type_info<fp>::type;
    int m_ref, n_ref, lda_ref, ldb_ref, group_size_ref;
    CBLAS_TRANSPOSE trans_ref;
    CBLAS_SIDE side_ref;
    CBLAS_DIAG diag_ref;
    CBLAS_UPLO uplo_ref;
    idx = 0;
    for (i = 0; i < group_count; i++) {
        m_ref          = (int)m[i];
        n_ref          = (int)n[i];
        lda_ref        = (int)lda[i];
        ldb_ref        = (int)ldb[i];
        group_size_ref = (int)group_size[i];
        trans_ref      = convert_to_cblas_trans(trans[i]);
        side_ref       = convert_to_cblas_side(left_right[i]);
        diag_ref       = convert_to_cblas_diag(unit_diag[i]);
        uplo_ref       = convert_to_cblas_uplo(upper_lower[i]);
        for (j = 0; j < group_size_ref; j++) {
            ::trsm(side_ref, uplo_ref, trans_ref, diag_ref, (const int *)&m_ref,
                   (const int *)&n_ref, (const fp_ref *)&alpha[i], (const fp_ref *)a_array[idx],
                   (const int *)&lda_ref, (fp_ref *)b_ref_array[idx], (const int *)&ldb_ref);
            idx++;
        }
    }

    // Call DPC++ TRSM_BATCH.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during TRSM_BATCH:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<onemkl::side, 1> side_buffer(left_right, range<1>(group_count));
    buffer<onemkl::uplo, 1> uplo_buffer(upper_lower, range<1>(group_count));
    buffer<onemkl::transpose, 1> trans_buffer(trans, range<1>(group_count));
    buffer<onemkl::diag, 1> diag_buffer(unit_diag, range<1>(group_count));
    buffer<int64_t, 1> m_buffer(m, range<1>(group_count));
    buffer<int64_t, 1> n_buffer(n, range<1>(group_count));
    buffer<int64_t, 1> lda_buffer(lda, range<1>(group_count));
    buffer<int64_t, 1> ldb_buffer(ldb, range<1>(group_count));
    buffer<int64_t, 1> group_size_buffer(group_size, range<1>(group_count));
    buffer<fp, 1> alpha_buffer(alpha, range<1>(group_count));
    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));

    try {
#ifdef CALL_RT_API
        onemkl::blas::trsm_batch(main_queue, side_buffer, uplo_buffer, trans_buffer, diag_buffer,
                                 m_buffer, n_buffer, alpha_buffer, A_buffer, lda_buffer, B_buffer,
                                 ldb_buffer, group_count, group_size_buffer);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::trsm_batch,
                    (main_queue, side_buffer, uplo_buffer, trans_buffer, diag_buffer, m_buffer,
                     n_buffer, alpha_buffer, A_buffer, lda_buffer, B_buffer, ldb_buffer,
                     group_count, group_size_buffer));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during TRSM_BATCH:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of TRSM_BATCH:\n" << error.what() << std::endl;
#ifdef ENABLE_CUBLAS_BACKEND
        // TRSM_BATCH currently not supported with CUBLAS backend.
        std::string error_msg(error.what());
        if (error_msg.compare("Not implemented for cublas") == 0) {
            onemkl::aligned_free(m);
            onemkl::aligned_free(n);
            onemkl::aligned_free(lda);
            onemkl::aligned_free(ldb);
            onemkl::aligned_free(trans);
            onemkl::aligned_free(left_right);
            onemkl::aligned_free(upper_lower);
            onemkl::aligned_free(unit_diag);
            onemkl::aligned_free(alpha);
            onemkl::aligned_free(group_size);
            onemkl::aligned_free(a_array);
            onemkl::aligned_free(b_array);
            onemkl::aligned_free(b_ref_array);
            return true;
        }
#endif
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto B_accessor = B_buffer.template get_access<access::mode::read>();
        good = check_equal_trsm_matrix(B_accessor, B_ref, total_size_b, 1, total_size_b, 10 * max,
                                       std::cout);
    }

    onemkl::aligned_free(m);
    onemkl::aligned_free(n);
    onemkl::aligned_free(lda);
    onemkl::aligned_free(ldb);
    onemkl::aligned_free(trans);
    onemkl::aligned_free(left_right);
    onemkl::aligned_free(upper_lower);
    onemkl::aligned_free(unit_diag);
    onemkl::aligned_free(alpha);
    onemkl::aligned_free(group_size);
    onemkl::aligned_free(a_array);
    onemkl::aligned_free(b_array);
    onemkl::aligned_free(b_ref_array);

    return good;
}

class TrsmBatchTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(TrsmBatchTests, RealSinglePrecision) {
    EXPECT_TRUE(test<float>(GetParam(), 5));
}

TEST_P(TrsmBatchTests, RealDoublePrecision) {
    EXPECT_TRUE(test<double>(GetParam(), 5));
}

TEST_P(TrsmBatchTests, ComplexSinglePrecision) {
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 5));
}

TEST_P(TrsmBatchTests, ComplexDoublePrecision) {
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 5));
}

INSTANTIATE_TEST_SUITE_P(TrsmBatchTestSuite, TrsmBatchTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
