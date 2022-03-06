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
#include "cblas.h"
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "allocator_helper.hpp"
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
int test(device *dev, oneapi::mkl::layout layout, int64_t group_count) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during TRSM_BATCH:\n"
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
    auto uaint = usm_allocator<int64_t, usm::alloc::shared, 64>(cxt, *dev);
    vector<int64_t, decltype(uaint)> m(uaint), n(uaint), lda(uaint), ldb(uaint), group_size(uaint);

    auto uatranspose = usm_allocator<oneapi::mkl::transpose, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::mkl::transpose, decltype(uatranspose)> trans(uatranspose);

    auto uaside = usm_allocator<oneapi::mkl::side, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::mkl::side, decltype(uaside)> left_right(uaside);

    auto uauplo = usm_allocator<oneapi::mkl::uplo, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::mkl::uplo, decltype(uauplo)> upper_lower(uauplo);

    auto uadiag = usm_allocator<oneapi::mkl::diag, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::mkl::diag, decltype(uadiag)> unit_nonunit(uadiag);

    auto uafp = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(uafp)> alpha(uafp);

    m.resize(group_count);
    n.resize(group_count);
    lda.resize(group_count);
    ldb.resize(group_count);
    group_size.resize(group_count);
    trans.resize(group_count);
    left_right.resize(group_count);
    upper_lower.resize(group_count);
    unit_nonunit.resize(group_count);
    alpha.resize(group_count);

    int64_t i, tmp;
    int64_t j, idx = 0;
    int64_t total_batch_count = 0;
    int64_t size_a = 0, size_b = 0;
    int64_t Arank = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        m[i] = 1 + std::rand() % 50;
        n[i] = 1 + std::rand() % 50;
        lda[i] = std::max(m[i], n[i]);
        ldb[i] = std::max(n[i], m[i]);
        alpha[i] = rand_scalar<fp>();
        if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
            trans[i] = (oneapi::mkl::transpose)(std::rand() % 2);
        }
        else {
            tmp = std::rand() % 3;
            if (tmp == 2)
                trans[i] = oneapi::mkl::transpose::conjtrans;
            else
                trans[i] = (oneapi::mkl::transpose)tmp;
        }
        left_right[i] = (oneapi::mkl::side)(std::rand() % 2);
        upper_lower[i] = (oneapi::mkl::uplo)(std::rand() % 2);
        unit_nonunit[i] = (oneapi::mkl::diag)(std::rand() % 2);

        total_batch_count += group_size[i];
    }

    auto uafpp = usm_allocator<fp *, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp *, decltype(uafpp)> a_array(uafpp), b_array(uafpp), b_ref_array(uafpp);

    a_array.resize(total_batch_count);
    b_array.resize(total_batch_count);
    b_ref_array.resize(total_batch_count);

    idx = 0;
    for (i = 0; i < group_count; i++) {
        size_a = lda[i] * (left_right[i] == oneapi::mkl::side::left ? m[i] : n[i]);
        Arank = left_right[i] == oneapi::mkl::side::left ? m[i] : n[i];
        size_b = ldb[i] * ((layout == oneapi::mkl::layout::column_major) ? n[i] : m[i]);
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx] = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_a, *dev, cxt);
            b_array[idx] = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_b, *dev, cxt);
            b_ref_array[idx] = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_b, *dev, cxt);
            rand_trsm_matrix(a_array[idx], layout, trans[i], Arank, Arank, lda[i]);
            rand_matrix(b_array[idx], layout, oneapi::mkl::transpose::nontrans, m[i], n[i], ldb[i]);
            copy_matrix(b_array[idx], layout, oneapi::mkl::transpose::nontrans, m[i], n[i], ldb[i],
                        b_ref_array[idx]);
            idx++;
        }
    }

    // Call reference TRSM_BATCH.
    using fp_ref = typename ref_type_info<fp>::type;
    int *m_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *n_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *lda_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *ldb_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int *group_size_ref = (int *)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);

    CBLAS_TRANSPOSE *trans_ref =
        (CBLAS_TRANSPOSE *)oneapi::mkl::aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);
    CBLAS_SIDE *left_right_ref =
        (CBLAS_SIDE *)oneapi::mkl::aligned_alloc(64, sizeof(CBLAS_SIDE) * group_count);
    CBLAS_UPLO *upper_lower_ref =
        (CBLAS_UPLO *)oneapi::mkl::aligned_alloc(64, sizeof(CBLAS_UPLO) * group_count);
    CBLAS_DIAG *unit_nonunit_ref =
        (CBLAS_DIAG *)oneapi::mkl::aligned_alloc(64, sizeof(CBLAS_DIAG) * group_count);

    if ((m_ref == NULL) || (n_ref == NULL) || (lda_ref == NULL) || (ldb_ref == NULL) ||
        (trans_ref == NULL) || (left_right_ref == NULL) || (upper_lower_ref == NULL) ||
        (unit_nonunit_ref == NULL) || (group_size_ref == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        oneapi::mkl::aligned_free(m_ref);
        oneapi::mkl::aligned_free(n_ref);
        oneapi::mkl::aligned_free(lda_ref);
        oneapi::mkl::aligned_free(ldb_ref);
        oneapi::mkl::aligned_free(trans_ref);
        oneapi::mkl::aligned_free(left_right_ref);
        oneapi::mkl::aligned_free(upper_lower_ref);
        oneapi::mkl::aligned_free(unit_nonunit_ref);
        oneapi::mkl::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(a_array[idx], cxt);
                oneapi::mkl::free_shared(b_array[idx], cxt);
                oneapi::mkl::free_shared(b_ref_array[idx], cxt);
                idx++;
            }
        }
        return false;
    }
    idx = 0;
    for (i = 0; i < group_count; i++) {
        trans_ref[i] = convert_to_cblas_trans(trans[i]);
        left_right_ref[i] = convert_to_cblas_side(left_right[i]);
        upper_lower_ref[i] = convert_to_cblas_uplo(upper_lower[i]);
        unit_nonunit_ref[i] = convert_to_cblas_diag(unit_nonunit[i]);
        m_ref[i] = (int)m[i];
        n_ref[i] = (int)n[i];
        lda_ref[i] = (int)lda[i];
        ldb_ref[i] = (int)ldb[i];
        group_size_ref[i] = (int)group_size[i];
        for (j = 0; j < group_size_ref[i]; j++) {
            ::trsm(convert_to_cblas_layout(layout), left_right_ref[i], upper_lower_ref[i],
                   trans_ref[i], unit_nonunit_ref[i], (const int *)&m_ref[i],
                   (const int *)&n_ref[i], (const fp_ref *)&alpha[i], (const fp_ref *)a_array[idx],
                   (const int *)&lda_ref[i], b_ref_array[idx], (const int *)&ldb_ref[i]);
            idx++;
        }
    }

    // Call DPC++ TRSM_BATCH.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::trsm_batch(
                    main_queue, &left_right[0], &upper_lower[0], &trans[0], &unit_nonunit[0], &m[0],
                    &n[0], &alpha[0], (const fp **)&a_array[0], &lda[0], &b_array[0], &ldb[0],
                    group_count, &group_size[0], dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::trsm_batch(
                    main_queue, &left_right[0], &upper_lower[0], &trans[0], &unit_nonunit[0], &m[0],
                    &n[0], &alpha[0], (const fp **)&a_array[0], &lda[0], &b_array[0], &ldb[0],
                    group_count, &group_size[0], dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::trsm_batch,
                                   &left_right[0], &upper_lower[0], &trans[0], &unit_nonunit[0],
                                   &m[0], &n[0], &alpha[0], (const fp **)&a_array[0], &lda[0],
                                   &b_array[0], &ldb[0], group_count, &group_size[0], dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::trsm_batch,
                                   &left_right[0], &upper_lower[0], &trans[0], &unit_nonunit[0],
                                   &m[0], &n[0], &alpha[0], (const fp **)&a_array[0], &lda[0],
                                   &b_array[0], &ldb[0], group_count, &group_size[0], dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during TRSM_BATCH:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        oneapi::mkl::aligned_free(m_ref);
        oneapi::mkl::aligned_free(n_ref);
        oneapi::mkl::aligned_free(lda_ref);
        oneapi::mkl::aligned_free(ldb_ref);
        oneapi::mkl::aligned_free(trans_ref);
        oneapi::mkl::aligned_free(left_right_ref);
        oneapi::mkl::aligned_free(upper_lower_ref);
        oneapi::mkl::aligned_free(unit_nonunit_ref);
        oneapi::mkl::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(a_array[idx], cxt);
                oneapi::mkl::free_shared(b_array[idx], cxt);
                oneapi::mkl::free_shared(b_ref_array[idx], cxt);
                idx++;
            }
        }
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of TRSM_BATCH:\n" << error.what() << std::endl;
    }

    bool good = true;
    // Compare the results of reference implementation and DPC++ implementation.
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            good = good && check_equal_trsm_matrix(b_array[idx], b_ref_array[idx], layout, m[i],
                                                   n[i], ldb[i], 10 * ldb[i], std::cout);
            idx++;
        }
    }
    oneapi::mkl::aligned_free(m_ref);
    oneapi::mkl::aligned_free(n_ref);
    oneapi::mkl::aligned_free(lda_ref);
    oneapi::mkl::aligned_free(ldb_ref);
    oneapi::mkl::aligned_free(trans_ref);
    oneapi::mkl::aligned_free(left_right_ref);
    oneapi::mkl::aligned_free(upper_lower_ref);
    oneapi::mkl::aligned_free(unit_nonunit_ref);
    oneapi::mkl::aligned_free(group_size_ref);
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            oneapi::mkl::free_shared(a_array[idx], cxt);
            oneapi::mkl::free_shared(b_array[idx], cxt);
            oneapi::mkl::free_shared(b_ref_array[idx], cxt);
            idx++;
        }
    }

    return (int)good;
}

class TrsmBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(TrsmBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(TrsmBatchUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(TrsmBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(TrsmBatchUsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(TrsmBatchUsmTestSuite, TrsmBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
