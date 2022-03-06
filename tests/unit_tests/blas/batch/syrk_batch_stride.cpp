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
int test(device *dev, oneapi::mkl::layout layout, int64_t batch_size) {
    // Prepare data.
    int64_t n, k;
    int64_t lda, ldc;
    oneapi::mkl::uplo upper_lower;
    oneapi::mkl::transpose trans;
    fp alpha, beta;
    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    n = 1 + std::rand() % 500;
    k = 1 + std::rand() % 500;
    lda = std::max(n, k);
    ldc = std::max(n, n);
    alpha = rand_scalar<fp>();
    beta = rand_scalar<fp>();

    upper_lower = (oneapi::mkl::uplo)(std::rand() % 2);
    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        trans = (std::rand() % 2) == 0 ? oneapi::mkl::transpose::nontrans
                                       : (std::rand() % 2) == 0 ? oneapi::mkl::transpose::trans
                                                                : oneapi::mkl::transpose::conjtrans;
    }
    else {
        trans = (std::rand() % 2) == 0 ? oneapi::mkl::transpose::nontrans
                                       : oneapi::mkl::transpose::trans;
    }

    int64_t stride_a, stride_c;

    switch (layout) {
        case oneapi::mkl::layout::column_major:
            stride_a = (trans == oneapi::mkl::transpose::nontrans) ? lda * k : lda * n;
            stride_c = ldc * n;
            break;
        case oneapi::mkl::layout::row_major:
            stride_a = (trans == oneapi::mkl::transpose::nontrans) ? lda * n : lda * k;
            stride_c = ldc * n;
            break;
        default: break;
    }

    vector<fp, allocator_helper<fp, 64>> A(stride_a * batch_size);
    vector<fp, allocator_helper<fp, 64>> C(stride_c * batch_size), C_ref(stride_c * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_matrix(A.data() + stride_a * i, layout, trans, n, k, lda);
        rand_matrix(C.data() + stride_c * i, layout, oneapi::mkl::transpose::nontrans, n, n, ldc);
    }

    C_ref = C;

    // Call reference SYRK_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int n_ref = (int)n;
    int k_ref = (int)k;
    int lda_ref = (int)lda;
    int ldc_ref = (int)ldc;
    int batch_size_ref = (int)batch_size;

    for (i = 0; i < batch_size_ref; i++) {
        ::syrk(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower),
               convert_to_cblas_trans(trans), (const int *)&n_ref, (const int *)&k_ref,
               (const fp_ref *)&alpha, (const fp_ref *)(A.data() + stride_a * i),
               (const int *)&lda_ref, (const fp_ref *)&beta,
               (fp_ref *)(C_ref.data() + stride_c * i), (const int *)&ldc_ref);
    }

    // Call DPC++ SYRK_BATCH_STRIDE.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during SYRK_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::syrk_batch(main_queue, upper_lower, trans, n, k,
                                                            alpha, A_buffer, lda, stride_a, beta,
                                                            C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::syrk_batch(main_queue, upper_lower, trans, n, k,
                                                         alpha, A_buffer, lda, stride_a, beta,
                                                         C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::syrk_batch,
                                   upper_lower, trans, n, k, alpha, A_buffer, lda, stride_a, beta,
                                   C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::syrk_batch,
                                   upper_lower, trans, n, k, alpha, A_buffer, lda, stride_a, beta,
                                   C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during SYRK_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of SYRK_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto C_accessor = C_buffer.template get_access<access::mode::read>();
    bool good =
        check_equal_matrix(C_accessor, C_ref, oneapi::mkl::layout::column_major,
                           stride_c * batch_size, 1, stride_c * batch_size, 10 * k, std::cout);

    return (int)good;
}

class SyrkBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(SyrkBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchStrideTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchStrideTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(SyrkBatchStrideTestSuite, SyrkBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
