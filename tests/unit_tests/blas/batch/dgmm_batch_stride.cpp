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
int test(device *dev, oneapi::mkl::layout layout, oneapi::mkl::side left_right, int64_t incx,
         int64_t batch_size) {
    // Prepare data.
    int64_t m, n;
    int64_t lda, ldc;
    int64_t i, tmp;

    batch_size = 15;
    m = 25;
    n = 30;
    lda = 38;
    ldc = 42;

    int x_len = (left_right == oneapi::mkl::side::right) ? n : m;

    int64_t stride_a, stride_x, stride_c;
    stride_x = x_len * std::abs(incx);
    stride_a = lda * std::max(m, n);
    stride_c = ldc * std::max(m, n);

    vector<fp, allocator_helper<fp, 64>> x(stride_x * batch_size), A(stride_a * batch_size);
    vector<fp, allocator_helper<fp, 64>> C(stride_c * batch_size), C_ref(stride_c * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_vector(x.data() + stride_x * i, x_len, incx);
        rand_matrix(A.data() + stride_a * i, layout, oneapi::mkl::transpose::nontrans, m, n, lda);
        rand_matrix(C.data() + stride_c * i, layout, oneapi::mkl::transpose::nontrans, m, n, ldc);
    }

    C_ref = C;

    // Call reference DGMM_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int m_ref = (int)m;
    int n_ref = (int)n;
    int incx_ref = (int)incx;
    int lda_ref = (int)lda;
    int ldc_ref = (int)ldc;
    int batch_size_ref = (int)batch_size;

    for (i = 0; i < batch_size_ref; i++) {
        ::dgmm(convert_to_cblas_layout(layout), convert_to_cblas_side(left_right),
               (const int *)&m_ref, (const int *)&n_ref, (const fp_ref *)(A.data() + stride_a * i),
               (const int *)&lda_ref, (const fp_ref *)(x.data() + stride_x * i),
               (const int *)&incx_ref, (fp_ref *)(C_ref.data() + stride_c * i),
               (const int *)&ldc_ref);
    }

    // Call DPC++ DGMM_BATCH_STRIDE.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during DGMM_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer(x.data(), range<1>(x.size()));
    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::dgmm_batch(main_queue, left_right, m, n, A_buffer,
                                                            lda, stride_a, x_buffer, incx, stride_x,
                                                            C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::dgmm_batch(main_queue, left_right, m, n, A_buffer,
                                                         lda, stride_a, x_buffer, incx, stride_x,
                                                         C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::dgmm_batch,
                                   left_right, m, n, A_buffer, lda, stride_a, x_buffer, incx,
                                   stride_x, C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::dgmm_batch, left_right,
                                   m, n, A_buffer, lda, stride_a, x_buffer, incx, stride_x,
                                   C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during DGMM_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of DGMM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto C_accessor = C_buffer.template get_access<access::mode::read>();
    bool good = true;
    for (i = 0; i < batch_size; i++) {
        good = good &&
               check_equal_matrix(C_accessor.get_pointer() + i * stride_c,
                                  C_ref.data() + i * stride_c, layout, m, n, ldc, 1, std::cout);
    }
    return (int)good;
}

class DgmmBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(DgmmBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::mkl::side::right, 2, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::mkl::side::right, -2, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::mkl::side::right, 1, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::mkl::side::left, 2, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::mkl::side::left, -2, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                  oneapi::mkl::side::left, 1, 5));
}

TEST_P(DgmmBatchStrideTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::mkl::side::right, 2, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::mkl::side::right, -2, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::mkl::side::right, 1, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::mkl::side::left, 2, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::mkl::side::left, -2, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                   oneapi::mkl::side::left, 1, 5));
}

TEST_P(DgmmBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::right, 2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::right, -2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::right, 1, 5));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::left, 2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::left, -2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                oneapi::mkl::side::left, 1, 5));
}

TEST_P(DgmmBatchStrideTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::right, 2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::right, -2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::right, 1, 5));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::left, 2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::left, -2, 5));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()),
                                                 oneapi::mkl::side::left, 1, 5));
}

INSTANTIATE_TEST_SUITE_P(DgmmBatchStrideTestSuite, DgmmBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
