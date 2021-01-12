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

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device *> devices;

namespace {

template <typename fp>
int test(device *dev, oneapi::mkl::layout layout, int64_t group_count) {
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

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto uaint = usm_allocator<int64_t, usm::alloc::shared, 64>(cxt, *dev);
    vector<int64_t, decltype(uaint)> m(uaint), n(uaint), k(uaint), lda(uaint), ldb(uaint),
        ldc(uaint), group_size(uaint);

    auto uatranspose = usm_allocator<oneapi::mkl::transpose, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::mkl::transpose, decltype(uatranspose)> transa(uatranspose), transb(uatranspose);

    auto uafp = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(uafp)> alpha(uafp), beta(uafp);

    m.resize(group_count);
    n.resize(group_count);
    k.resize(group_count);
    lda.resize(group_count);
    ldb.resize(group_count);
    ldc.resize(group_count);
    group_size.resize(group_count);
    transa.resize(group_count);
    transb.resize(group_count);
    alpha.resize(group_count);
    beta.resize(group_count);

    int64_t i, tmp;
    int64_t j, idx = 0;
    int64_t total_batch_count = 0;
    int64_t size_a = 0, size_b = 0, size_c = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        m[i] = 1 + std::rand() % 500;
        n[i] = 1 + std::rand() % 500;
        k[i] = 1 + std::rand() % 500;
        lda[i] = std::max(m[i], k[i]);
        ldb[i] = std::max(n[i], k[i]);
        ldc[i] = std::max(m[i], n[i]);
        alpha[i] = rand_scalar<fp>();
        beta[i] = rand_scalar<fp>();
        if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
            transa[i] = (oneapi::mkl::transpose)(std::rand() % 2);
            transb[i] = (oneapi::mkl::transpose)(std::rand() % 2);
        }
        else {
            tmp = std::rand() % 3;
            if (tmp == 2)
                transa[i] = oneapi::mkl::transpose::conjtrans;
            else
                transa[i] = (oneapi::mkl::transpose)tmp;
            tmp = std::rand() % 3;
            if (tmp == 2)
                transb[i] = oneapi::mkl::transpose::conjtrans;
            else
                transb[i] = (oneapi::mkl::transpose)tmp;
        }
        total_batch_count += group_size[i];
    }

    auto uafpp = usm_allocator<fp *, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp *, decltype(uafpp)> a_array(uafpp), b_array(uafpp), c_array(uafpp),
        c_ref_array(uafpp);
    a_array.resize(total_batch_count);
    b_array.resize(total_batch_count);
    c_array.resize(total_batch_count);
    c_ref_array.resize(total_batch_count);

    idx = 0;
    for (i = 0; i < group_count; i++) {
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                size_a = lda[i] * ((transa[i] == oneapi::mkl::transpose::nontrans) ? k[i] : m[i]);
                size_b = ldb[i] * ((transb[i] == oneapi::mkl::transpose::nontrans) ? n[i] : k[i]);
                size_c = ldc[i] * n[i];
                break;
            case oneapi::mkl::layout::row_major:
                size_a = lda[i] * ((transa[i] == oneapi::mkl::transpose::nontrans) ? m[i] : k[i]);
                size_b = ldb[i] * ((transb[i] == oneapi::mkl::transpose::nontrans) ? k[i] : n[i]);
                size_c = ldc[i] * m[i];
                break;
            default: break;
        }
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx] = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_a, *dev, cxt);
            b_array[idx] = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_b, *dev, cxt);
            c_array[idx] = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_c, *dev, cxt);
            c_ref_array[idx] = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_c, *dev, cxt);
            rand_matrix(a_array[idx], layout, transa[i], m[i], k[i], lda[i]);
            rand_matrix(b_array[idx], layout, transb[i], k[i], n[i], ldb[i]);
            rand_matrix(c_array[idx], layout, oneapi::mkl::transpose::nontrans, m[i], n[i], ldc[i]);
            copy_matrix(c_array[idx], layout, oneapi::mkl::transpose::nontrans, m[i], n[i], ldc[i],
                        c_ref_array[idx]);
            idx++;
        }
    }

    // Call reference GEMM_BATCH.
    using fp_ref = typename ref_type_info<fp>::type;
    int *m_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *n_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *k_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *lda_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *ldb_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *ldc_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *group_size_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);

    CBLAS_TRANSPOSE *transa_ref =
        (CBLAS_TRANSPOSE *)oneapi::mkl::aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);
    CBLAS_TRANSPOSE *transb_ref =
        (CBLAS_TRANSPOSE *)oneapi::mkl::aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);

    if ((m_ref == NULL) || (n_ref == NULL) || (k_ref == NULL) || (lda_ref == NULL) ||
        (ldb_ref == NULL) || (ldc_ref == NULL) || (transa_ref == NULL) || (transb_ref == NULL) ||
        (group_size_ref == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        oneapi::mkl::aligned_free(m_ref);
        oneapi::mkl::aligned_free(n_ref);
        oneapi::mkl::aligned_free(k_ref);
        oneapi::mkl::aligned_free(lda_ref);
        oneapi::mkl::aligned_free(ldb_ref);
        oneapi::mkl::aligned_free(ldc_ref);
        oneapi::mkl::aligned_free(transa_ref);
        oneapi::mkl::aligned_free(transb_ref);
        oneapi::mkl::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(a_array[idx], cxt);
                oneapi::mkl::free_shared(b_array[idx], cxt);
                oneapi::mkl::free_shared(c_array[idx], cxt);
                oneapi::mkl::free_shared(c_ref_array[idx], cxt);
                idx++;
            }
        }
        return false;
    }
    idx = 0;
    for (i = 0; i < group_count; i++) {
        transa_ref[i] = convert_to_cblas_trans(transa[i]);
        transb_ref[i] = convert_to_cblas_trans(transb[i]);
        m_ref[i] = (int)m[i];
        n_ref[i] = (int)n[i];
        k_ref[i] = (int)k[i];
        lda_ref[i] = (int)lda[i];
        ldb_ref[i] = (int)ldb[i];
        ldc_ref[i] = (int)ldc[i];
        group_size_ref[i] = (int)group_size[i];
        for (j = 0; j < group_size_ref[i]; j++) {
            ::gemm(convert_to_cblas_layout(layout), transa_ref[i], transb_ref[i],
                   (const int *)&m_ref[i], (const int *)&n_ref[i], (const int *)&k_ref[i],
                   (const fp_ref *)&alpha[i], (const fp_ref *)a_array[idx],
                   (const int *)&lda_ref[i], (const fp_ref *)b_array[idx], (const int *)&ldb_ref[i],
                   (const fp_ref *)&beta[i], (fp_ref *)c_ref_array[idx], (const int *)&ldc_ref[i]);
            idx++;
        }
    }

    // Call DPC++ GEMM_BATCH.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::gemm_batch(
                    main_queue, &transa[0], &transb[0], &m[0], &n[0], &k[0], &alpha[0],
                    (const fp **)&a_array[0], &lda[0], (const fp **)&b_array[0], &ldb[0], &beta[0],
                    &c_array[0], &ldc[0], group_count, &group_size[0], dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::gemm_batch(
                    main_queue, &transa[0], &transb[0], &m[0], &n[0], &k[0], &alpha[0],
                    (const fp **)&a_array[0], &lda[0], (const fp **)&b_array[0], &ldb[0], &beta[0],
                    &c_array[0], &ldc[0], group_count, &group_size[0], dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::gemm_batch,
                                   &transa[0], &transb[0], &m[0], &n[0], &k[0], &alpha[0],
                                   (const fp **)&a_array[0], &lda[0], (const fp **)&b_array[0],
                                   &ldb[0], &beta[0], &c_array[0], &ldc[0], group_count,
                                   &group_size[0], dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::gemm_batch, &transa[0],
                                   &transb[0], &m[0], &n[0], &k[0], &alpha[0],
                                   (const fp **)&a_array[0], &lda[0], (const fp **)&b_array[0],
                                   &ldb[0], &beta[0], &c_array[0], &ldc[0], group_count,
                                   &group_size[0], dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented &e) {
        oneapi::mkl::aligned_free(m_ref);
        oneapi::mkl::aligned_free(n_ref);
        oneapi::mkl::aligned_free(k_ref);
        oneapi::mkl::aligned_free(lda_ref);
        oneapi::mkl::aligned_free(ldb_ref);
        oneapi::mkl::aligned_free(ldc_ref);
        oneapi::mkl::aligned_free(transa_ref);
        oneapi::mkl::aligned_free(transb_ref);
        oneapi::mkl::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(a_array[idx], cxt);
                oneapi::mkl::free_shared(b_array[idx], cxt);
                oneapi::mkl::free_shared(c_array[idx], cxt);
                oneapi::mkl::free_shared(c_ref_array[idx], cxt);
                idx++;
            }
        }
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GEMM_BATCH:\n" << error.what() << std::endl;
    }

    bool good = true;
    // Compare the results of reference implementation and DPC++ implementation.
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            good = good && check_equal_matrix(c_array[idx], c_ref_array[idx], layout, m[i], n[i],
                                              ldc[i], 10 * k[i], std::cout);
            idx++;
        }
    }
    oneapi::mkl::aligned_free(m_ref);
    oneapi::mkl::aligned_free(n_ref);
    oneapi::mkl::aligned_free(k_ref);
    oneapi::mkl::aligned_free(lda_ref);
    oneapi::mkl::aligned_free(ldb_ref);
    oneapi::mkl::aligned_free(ldc_ref);
    oneapi::mkl::aligned_free(transa_ref);
    oneapi::mkl::aligned_free(transb_ref);
    oneapi::mkl::aligned_free(group_size_ref);
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            oneapi::mkl::free_shared(a_array[idx], cxt);
            oneapi::mkl::free_shared(b_array[idx], cxt);
            oneapi::mkl::free_shared(c_array[idx], cxt);
            oneapi::mkl::free_shared(c_ref_array[idx], cxt);
            idx++;
        }
    }

    return (int)good;
}

class GemmBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device *, oneapi::mkl::layout>> {};

TEST_P(GemmBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(GemmBatchUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(GemmBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(GemmBatchUsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(GemmBatchUsmTestSuite, GemmBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
