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
int test(const device &dev, int64_t group_count) {
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
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    int64_t *m   = (int64_t *)onemkl::malloc_shared(64, sizeof(int64_t) * group_count, dev, cxt);
    int64_t *n   = (int64_t *)onemkl::malloc_shared(64, sizeof(int64_t) * group_count, dev, cxt);
    int64_t *k   = (int64_t *)onemkl::malloc_shared(64, sizeof(int64_t) * group_count, dev, cxt);
    int64_t *lda = (int64_t *)onemkl::malloc_shared(64, sizeof(int64_t) * group_count, dev, cxt);
    int64_t *ldb = (int64_t *)onemkl::malloc_shared(64, sizeof(int64_t) * group_count, dev, cxt);
    int64_t *ldc = (int64_t *)onemkl::malloc_shared(64, sizeof(int64_t) * group_count, dev, cxt);
    onemkl::transpose *transa = (onemkl::transpose *)onemkl::malloc_shared(
        64, sizeof(onemkl::transpose) * group_count, dev, cxt);
    onemkl::transpose *transb = (onemkl::transpose *)onemkl::malloc_shared(
        64, sizeof(onemkl::transpose) * group_count, dev, cxt);
    fp *alpha = (fp *)onemkl::malloc_shared(64, sizeof(fp) * group_count, dev, cxt);
    fp *beta  = (fp *)onemkl::malloc_shared(64, sizeof(fp) * group_count, dev, cxt);
    int64_t *group_size =
        (int64_t *)onemkl::malloc_shared(64, sizeof(int64_t) * group_count, dev, cxt);

    if ((m == NULL) || (n == NULL) || (k == NULL) || (lda == NULL) || (ldb == NULL) ||
        (ldc == NULL) || (transa == NULL) || (transb == NULL) || (alpha == NULL) ||
        (beta == NULL) || (group_size == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        if (!dev.is_host()) {
            onemkl::free_shared(m, cxt);
            onemkl::free_shared(n, cxt);
            onemkl::free_shared(k, cxt);
            onemkl::free_shared(lda, cxt);
            onemkl::free_shared(ldb, cxt);
            onemkl::free_shared(ldc, cxt);
            onemkl::free_shared(transa, cxt);
            onemkl::free_shared(transb, cxt);
            onemkl::free_shared(alpha, cxt);
            onemkl::free_shared(beta, cxt);
            onemkl::free_shared(group_size, cxt);
        }
        return false;
    }

    int64_t i, tmp;
    int64_t j, idx = 0;
    int64_t total_batch_count = 0;

    int64_t *total_size_a = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *total_size_b = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    int64_t *total_size_c = (int64_t *)onemkl::aligned_alloc(64, sizeof(int64_t) * group_count);
    if ((total_size_a == NULL) || (total_size_b == NULL) || (total_size_c == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        onemkl::aligned_free(total_size_a);
        onemkl::aligned_free(total_size_b);
        onemkl::aligned_free(total_size_c);
        return false;
    }

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
        total_size_a[i] = lda[i] * ((transa[i] == onemkl::transpose::nontrans) ? k[i] : m[i]);
        total_size_b[i] = ldb[i] * ((transb[i] == onemkl::transpose::nontrans) ? n[i] : k[i]);
        total_size_c[i] = ldc[i] * n[i];
        total_batch_count += group_size[i];
    }

    fp **a_array     = (fp **)onemkl::malloc_shared(64, sizeof(fp *) * total_batch_count, dev, cxt);
    fp **b_array     = (fp **)onemkl::malloc_shared(64, sizeof(fp *) * total_batch_count, dev, cxt);
    fp **c_array     = (fp **)onemkl::malloc_shared(64, sizeof(fp *) * total_batch_count, dev, cxt);
    fp **c_ref_array = (fp **)onemkl::malloc_shared(64, sizeof(fp *) * total_batch_count, dev, cxt);

    if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL) || (c_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        if (!dev.is_host()) {
            onemkl::free_shared(a_array, cxt);
            onemkl::free_shared(b_array, cxt);
            onemkl::free_shared(c_array, cxt);
            onemkl::free_shared(c_ref_array, cxt);
        }
        return false;
    }
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx] = (fp *)onemkl::malloc_shared(64, sizeof(fp) * total_size_a[i], dev, cxt);
            b_array[idx] = (fp *)onemkl::malloc_shared(64, sizeof(fp) * total_size_b[i], dev, cxt);
            c_array[idx] = (fp *)onemkl::malloc_shared(64, sizeof(fp) * total_size_c[i], dev, cxt);
            c_ref_array[idx] =
                (fp *)onemkl::malloc_shared(64, sizeof(fp) * total_size_c[i], dev, cxt);

            rand_matrix(a_array[idx], transa[i], m[i], k[i], lda[i]);
            rand_matrix(b_array[idx], transb[i], k[i], n[i], ldb[i]);
            rand_matrix(c_array[idx], onemkl::transpose::nontrans, m[i], n[i], ldc[i]);
            copy_matrix(c_array[idx], onemkl::transpose::nontrans, m[i], n[i], ldc[i],
                        c_ref_array[idx]);
            idx++;
        }
    }

    // Call reference GEMM_BATCH.
    using fp_ref        = typename ref_type_info<fp>::type;
    int *m_ref          = (int *)onemkl::aligned_alloc(64, sizeof(int) * group_count);
    int *n_ref          = (int *)onemkl::aligned_alloc(64, sizeof(int) * group_count);
    int *k_ref          = (int *)onemkl::aligned_alloc(64, sizeof(int) * group_count);
    int *lda_ref        = (int *)onemkl::aligned_alloc(64, sizeof(int) * group_count);
    int *ldb_ref        = (int *)onemkl::aligned_alloc(64, sizeof(int) * group_count);
    int *ldc_ref        = (int *)onemkl::aligned_alloc(64, sizeof(int) * group_count);
    int *group_size_ref = (int *)onemkl::aligned_alloc(64, sizeof(int) * group_count);

    CBLAS_TRANSPOSE *transa_ref =
        (CBLAS_TRANSPOSE *)aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);
    CBLAS_TRANSPOSE *transb_ref =
        (CBLAS_TRANSPOSE *)aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);

    if ((m_ref == NULL) || (n_ref == NULL) || (k_ref == NULL) || (lda_ref == NULL) ||
        (ldb_ref == NULL) || (ldc_ref == NULL) || (transa_ref == NULL) || (transb_ref == NULL) ||
        (group_size_ref == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        onemkl::aligned_free(m_ref);
        onemkl::aligned_free(n_ref);
        onemkl::aligned_free(k_ref);
        onemkl::aligned_free(lda_ref);
        onemkl::aligned_free(ldb_ref);
        onemkl::aligned_free(ldc_ref);
        onemkl::aligned_free(transa_ref);
        onemkl::aligned_free(transb_ref);
        onemkl::aligned_free(group_size_ref);
        return false;
    }
    idx = 0;
    for (i = 0; i < group_count; i++) {
        transa_ref[i]     = convert_to_cblas_trans(transa[i]);
        transb_ref[i]     = convert_to_cblas_trans(transb[i]);
        m_ref[i]          = (int)m[i];
        n_ref[i]          = (int)n[i];
        k_ref[i]          = (int)k[i];
        lda_ref[i]        = (int)lda[i];
        ldb_ref[i]        = (int)ldb[i];
        ldc_ref[i]        = (int)ldc[i];
        group_size_ref[i] = (int)group_size[i];
        for (j = 0; j < group_size_ref[i]; j++) {
            ::gemm(transa_ref[i], transb_ref[i], (const int *)&m_ref[i], (const int *)&n_ref[i],
                   (const int *)&k_ref[i], (const fp_ref *)&alpha[i], (const fp_ref *)a_array[idx],
                   (const int *)&lda_ref[i], (const fp_ref *)b_array[idx], (const int *)&ldb_ref[i],
                   (const fp_ref *)&beta[i], (fp_ref *)c_ref_array[idx], (const int *)&ldc_ref[i]);
            idx++;
        }
    }

    // Call DPC++ GEMM_BATCH.

    try {
#ifdef CALL_RT_API
        done = onemkl::blas::gemm_batch(main_queue, transa, transb, m, n, k, alpha,
                                        (const fp **)a_array, lda, (const fp **)b_array, ldb, beta,
                                        c_array, ldc, group_count, group_size, dependencies);
        done.wait();
#else
        TEST_RUN_CT(
            main_queue, onemkl::blas::gemm_batch,
            (main_queue, transa, transb, m, n, k, alpha, (const fp **)a_array, lda,
             (const fp **)b_array, ldb, beta, c_array, ldc, group_count, group_size, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const backend_unsupported_exception &e) {
        onemkl::aligned_free(total_size_a);
        onemkl::aligned_free(total_size_b);
        onemkl::aligned_free(total_size_c);
        onemkl::aligned_free(m_ref);
        onemkl::aligned_free(n_ref);
        onemkl::aligned_free(k_ref);
        onemkl::aligned_free(lda_ref);
        onemkl::aligned_free(ldb_ref);
        onemkl::aligned_free(ldc_ref);
        onemkl::aligned_free(transa_ref);
        onemkl::aligned_free(transb_ref);
        onemkl::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                if (!dev.is_host()) {
                    onemkl::free_shared(a_array[idx], cxt);
                    onemkl::free_shared(b_array[idx], cxt);
                    onemkl::free_shared(c_array[idx], cxt);
                    onemkl::free_shared(c_ref_array[idx], cxt);
                }
                idx++;
            }
        }
        if (!dev.is_host()) {
            onemkl::free_shared(m, cxt);
            onemkl::free_shared(n, cxt);
            onemkl::free_shared(k, cxt);
            onemkl::free_shared(lda, cxt);
            onemkl::free_shared(ldb, cxt);
            onemkl::free_shared(ldc, cxt);
            onemkl::free_shared(transa, cxt);
            onemkl::free_shared(transb, cxt);
            onemkl::free_shared(alpha, cxt);
            onemkl::free_shared(beta, cxt);
            onemkl::free_shared(group_size, cxt);
            onemkl::free_shared(a_array, cxt);
            onemkl::free_shared(b_array, cxt);
            onemkl::free_shared(c_array, cxt);
            onemkl::free_shared(c_ref_array, cxt);
        }
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GEMM_BATCH:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = true;
    {
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                good = good && check_equal_matrix(c_array[idx], c_ref_array[idx], m[i], n[i],
                                                  ldc[i], 10 * k[i], std::cout);
                idx++;
            }
        }
    }

    onemkl::aligned_free(total_size_a);
    onemkl::aligned_free(total_size_b);
    onemkl::aligned_free(total_size_c);
    onemkl::aligned_free(m_ref);
    onemkl::aligned_free(n_ref);
    onemkl::aligned_free(k_ref);
    onemkl::aligned_free(lda_ref);
    onemkl::aligned_free(ldb_ref);
    onemkl::aligned_free(ldc_ref);
    onemkl::aligned_free(transa_ref);
    onemkl::aligned_free(transb_ref);
    onemkl::aligned_free(group_size_ref);
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            if (!dev.is_host()) {
                onemkl::free_shared(a_array[idx], cxt);
                onemkl::free_shared(b_array[idx], cxt);
                onemkl::free_shared(c_array[idx], cxt);
                onemkl::free_shared(c_ref_array[idx], cxt);
            }
            idx++;
        }
    }
    if (!dev.is_host()) {
        onemkl::free_shared(m, cxt);
        onemkl::free_shared(n, cxt);
        onemkl::free_shared(k, cxt);
        onemkl::free_shared(lda, cxt);
        onemkl::free_shared(ldb, cxt);
        onemkl::free_shared(ldc, cxt);
        onemkl::free_shared(transa, cxt);
        onemkl::free_shared(transb, cxt);
        onemkl::free_shared(alpha, cxt);
        onemkl::free_shared(beta, cxt);
        onemkl::free_shared(group_size, cxt);
        onemkl::free_shared(a_array, cxt);
        onemkl::free_shared(b_array, cxt);
        onemkl::free_shared(c_array, cxt);
        onemkl::free_shared(c_ref_array, cxt);
    }
    return (int)good;
}

class GemmBatchUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GemmBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(GetParam(), 5));
}

TEST_P(GemmBatchUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(GetParam(), 5));
}

TEST_P(GemmBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), 5));
}

TEST_P(GemmBatchUsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), 5));
}

INSTANTIATE_TEST_SUITE_P(GemmBatchUsmTestSuite, GemmBatchUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
