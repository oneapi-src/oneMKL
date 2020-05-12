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
    int64_t *k   = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *lda = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *ldb = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *ldc = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    onemkl::transpose *transa =
        (onemkl::transpose *)onemkl::aligned_alloc(64, sizeof(onemkl::transpose) * group_count);
    onemkl::transpose *transb =
        (onemkl::transpose *)onemkl::aligned_alloc(64, sizeof(onemkl::transpose) * group_count);
    fp *alpha           = (fp *)onemkl::aligned_alloc(64, sizeof(fp) * group_count);
    fp *beta            = (fp *)onemkl::aligned_alloc(64, sizeof(fp) * group_count);
    int64_t *group_size = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);

    if ((m == NULL) || (n == NULL) || (k == NULL) || (lda == NULL) || (ldb == NULL) ||
        (ldc == NULL) || (transa == NULL) || (transb == NULL) || (alpha == NULL) ||
        (beta == NULL) || (group_size == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        onemkl::aligned_free(m);
        onemkl::aligned_free(n);
        onemkl::aligned_free(k);
        onemkl::aligned_free(lda);
        onemkl::aligned_free(ldb);
        onemkl::aligned_free(ldc);
        onemkl::aligned_free(transa);
        onemkl::aligned_free(transb);
        onemkl::aligned_free(alpha);
        onemkl::aligned_free(beta);
        onemkl::aligned_free(group_size);
        return false;
    }

    int64_t i, tmp;
    int64_t j, idx = 0, max_k = 0;
    int64_t total_size_a = 0, total_size_b = 0, total_size_c = 0, total_batch_count = 0;
    int64_t size_a = 0, size_b = 0, size_c = 0;
    int64_t off_a = 0, off_b = 0, off_c = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        m[i]          = 1 + std::rand() % 500;
        n[i]          = 1 + std::rand() % 500;
        k[i]          = 1 + std::rand() % 500;
        lda[i]        = std::max(m[i], k[i]);
        ldb[i]        = std::max(n[i], k[i]);
        ldc[i]        = std::max(m[i], n[i]);
        alpha[i]      = rand_scalar<fp>();
        beta[i]       = rand_scalar<fp>();
        if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
            transa[i] = (onemkl::transpose)(std::rand() % 2);
            transb[i] = (onemkl::transpose)(std::rand() % 2);
        }
        else {
            tmp = std::rand() % 3;
            if (tmp == 2)
                transa[i] = onemkl::transpose::conjtrans;
            else
                transa[i] = (onemkl::transpose)tmp;
            tmp = std::rand() % 3;
            if (tmp == 2)
                transb[i] = onemkl::transpose::conjtrans;
            else
                transb[i] = (onemkl::transpose)tmp;
        }
        total_size_a +=
            lda[i] * group_size[i] * ((transa[i] == onemkl::transpose::nontrans) ? k[i] : m[i]);
        total_size_b +=
            ldb[i] * group_size[i] * ((transb[i] == onemkl::transpose::nontrans) ? n[i] : k[i]);
        total_size_c += ldc[i] * n[i] * group_size[i];
        total_batch_count += group_size[i];
    }

    fp **a_array     = (fp **)onemkl::aligned_alloc(64, sizeof(fp *) * total_batch_count);
    fp **b_array     = (fp **)onemkl::aligned_alloc(64, sizeof(fp *) * total_batch_count);
    fp **c_array     = (fp **)onemkl::aligned_alloc(64, sizeof(fp *) * total_batch_count);
    fp **c_ref_array = (fp **)onemkl::aligned_alloc(64, sizeof(fp *) * total_batch_count);

    if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL) || (c_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        onemkl::aligned_free(a_array);
        onemkl::aligned_free(b_array);
        onemkl::aligned_free(c_array);
        onemkl::aligned_free(c_ref_array);
        return false;
    }

    vector<fp, allocator_helper<fp, 64>> A(total_size_a), B(total_size_b), C(total_size_c),
        C_ref(total_size_c);

    for (i = 0; i < group_count; i++) {
        max_k  = std::max(max_k, k[i]);
        size_a = (transa[i] == onemkl::transpose::nontrans) ? k[i] * lda[i] : m[i] * lda[i];
        size_b = (transb[i] == onemkl::transpose::nontrans) ? n[i] * ldb[i] : k[i] * ldb[i];
        size_c = n[i] * ldc[i];
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx]     = A.data() + off_a;
            b_array[idx]     = B.data() + off_b;
            c_array[idx]     = C.data() + off_c;
            c_ref_array[idx] = C_ref.data() + off_c;
            rand_matrix(a_array[idx], transa[i], m[i], k[i], lda[i]);
            rand_matrix(b_array[idx], transb[i], k[i], n[i], ldb[i]);
            rand_matrix(c_array[idx], onemkl::transpose::nontrans, m[i], n[i], ldc[i]);
            off_a += size_a;
            off_b += size_b;
            off_c += size_c;
            idx++;
        }
    }
    C_ref = C;

    // Call reference GEMM_BATCH.
    using fp_ref = typename ref_type_info<fp>::type;
    int m_ref, n_ref, k_ref, lda_ref, ldb_ref, ldc_ref, group_size_ref;
    CBLAS_TRANSPOSE transa_ref, transb_ref;
    idx = 0;
    for (i = 0; i < group_count; i++) {
        m_ref          = (int)m[i];
        n_ref          = (int)n[i];
        k_ref          = (int)k[i];
        lda_ref        = (int)lda[i];
        ldb_ref        = (int)ldb[i];
        ldc_ref        = (int)ldc[i];
        group_size_ref = (int)group_size[i];
        transa_ref     = convert_to_cblas_trans(transa[i]);
        transb_ref     = convert_to_cblas_trans(transb[i]);
        for (j = 0; j < group_size_ref; j++) {
            ::gemm(transa_ref, transb_ref, (const int *)&m_ref, (const int *)&n_ref,
                   (const int *)&k_ref, (const fp_ref *)&alpha[i], (const fp_ref *)a_array[idx],
                   (const int *)&lda_ref, (const fp_ref *)b_array[idx], (const int *)&ldb_ref,
                   (const fp_ref *)&beta[i], (fp_ref *)c_ref_array[idx], (const int *)&ldc_ref);
            idx++;
        }
    }

    // Call DPC++ GEMM_BATCH.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<onemkl::transpose, 1> transa_buffer(transa, range<1>(group_count));
    buffer<onemkl::transpose, 1> transb_buffer(transb, range<1>(group_count));
    buffer<std::int64_t, 1> m_buffer(m, range<1>(group_count));
    buffer<std::int64_t, 1> n_buffer(n, range<1>(group_count));
    buffer<std::int64_t, 1> k_buffer(k, range<1>(group_count));
    buffer<std::int64_t, 1> lda_buffer(lda, range<1>(group_count));
    buffer<std::int64_t, 1> ldb_buffer(ldb, range<1>(group_count));
    buffer<std::int64_t, 1> ldc_buffer(ldc, range<1>(group_count));
    buffer<std::int64_t, 1> group_size_buffer(group_size, range<1>(group_count));
    buffer<fp, 1> alpha_buffer(alpha, range<1>(group_count));
    buffer<fp, 1> beta_buffer(beta, range<1>(group_count));
    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        onemkl::blas::gemm_batch(main_queue, transa_buffer, transb_buffer, m_buffer, n_buffer,
                                 k_buffer, alpha_buffer, A_buffer, lda_buffer, B_buffer, ldb_buffer,
                                 beta_buffer, C_buffer, ldc_buffer, group_count, group_size_buffer);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::gemm_batch,
                    (main_queue, transa_buffer, transb_buffer, m_buffer, n_buffer, k_buffer,
                     alpha_buffer, A_buffer, lda_buffer, B_buffer, ldb_buffer, beta_buffer,
                     C_buffer, ldc_buffer, group_count, group_size_buffer));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GEMM_BATCH:\n" << error.what() << std::endl;
#ifdef ENABLE_CUBLAS_BACKEND
        // GEMM_BATCH currently not supported with CUBLAS backend.
        std::string error_msg(error.what());
        if (error_msg.compare("Not implemented for cublas") == 0) {
            onemkl::aligned_free(m);
            onemkl::aligned_free(n);
            onemkl::aligned_free(k);
            onemkl::aligned_free(lda);
            onemkl::aligned_free(ldb);
            onemkl::aligned_free(ldc);
            onemkl::aligned_free(transa);
            onemkl::aligned_free(transb);
            onemkl::aligned_free(alpha);
            onemkl::aligned_free(beta);
            onemkl::aligned_free(group_size);
            onemkl::aligned_free(a_array);
            onemkl::aligned_free(b_array);
            onemkl::aligned_free(c_array);
            onemkl::aligned_free(c_ref_array);
            return true;
        }
#endif
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto C_accessor = C_buffer.template get_access<access::mode::read>();
        good = check_equal_matrix(C_accessor, C_ref, total_size_c, 1, total_size_c, 10 * max_k,
                                  std::cout);
    }

    onemkl::aligned_free(m);
    onemkl::aligned_free(n);
    onemkl::aligned_free(k);
    onemkl::aligned_free(lda);
    onemkl::aligned_free(ldb);
    onemkl::aligned_free(ldc);
    onemkl::aligned_free(transa);
    onemkl::aligned_free(transb);
    onemkl::aligned_free(alpha);
    onemkl::aligned_free(beta);
    onemkl::aligned_free(group_size);
    onemkl::aligned_free(a_array);
    onemkl::aligned_free(b_array);
    onemkl::aligned_free(c_array);
    onemkl::aligned_free(c_ref_array);

    return good;
}

class GemmBatchTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GemmBatchTests, RealSinglePrecision) {
    EXPECT_TRUE(test<float>(GetParam(), 5));
}

TEST_P(GemmBatchTests, RealDoublePrecision) {
    EXPECT_TRUE(test<double>(GetParam(), 5));
}

TEST_P(GemmBatchTests, ComplexSinglePrecision) {
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 5));
}

TEST_P(GemmBatchTests, ComplexDoublePrecision) {
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 5));
}

INSTANTIATE_TEST_SUITE_P(GemmBatchTestSuite, GemmBatchTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
