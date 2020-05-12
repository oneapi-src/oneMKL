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
bool test(const device &dev, int64_t batch_size) {
    // Prepare data.
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    onemkl::transpose transa, transb;
    fp alpha, beta;
    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m          = 1 + std::rand() % 500;
    n          = 1 + std::rand() % 500;
    k          = 1 + std::rand() % 500;
    lda        = std::max(m, k);
    ldb        = std::max(n, k);
    ldc        = std::max(m, n);
    alpha      = rand_scalar<fp>();
    beta       = rand_scalar<fp>();

    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        transa = (onemkl::transpose)(std::rand() % 2);
        transb = (onemkl::transpose)(std::rand() % 2);
    }
    else {
        tmp = std::rand() % 3;
        if (tmp == 2)
            transa = onemkl::transpose::conjtrans;
        else
            transa = (onemkl::transpose)tmp;
        tmp = std::rand() % 3;
        if (tmp == 2)
            transb = onemkl::transpose::conjtrans;
        else
            transb = (onemkl::transpose)tmp;
    }

    int64_t stride_a, stride_b, stride_c;

    stride_a = (transa == onemkl::transpose::nontrans) ? lda * k : lda * m;
    stride_b = (transb == onemkl::transpose::nontrans) ? ldb * n : ldb * k;
    stride_c = ldc * n;

    vector<fp, allocator_helper<fp, 64>> A(stride_a * batch_size), B(stride_b * batch_size);
    vector<fp, allocator_helper<fp, 64>> C(stride_c * batch_size), C_ref(stride_c * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_matrix(A.data() + stride_a * i, transa, m, k, lda);
        rand_matrix(B.data() + stride_b * i, transb, k, n, ldb);
        rand_matrix(C.data() + stride_c * i, onemkl::transpose::nontrans, m, n, ldc);
    }

    C_ref = C;

    // Call reference GEMM_BATCH_STRIDE.
    using fp_ref       = typename ref_type_info<fp>::type;
    int m_ref          = (int)m;
    int n_ref          = (int)n;
    int k_ref          = (int)k;
    int lda_ref        = (int)lda;
    int ldb_ref        = (int)ldb;
    int ldc_ref        = (int)ldc;
    int batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        ::gemm(convert_to_cblas_trans(transa), convert_to_cblas_trans(transb), (const int *)&m_ref,
               (const int *)&n_ref, (const int *)&k_ref, (const fp_ref *)&alpha,
               (const fp_ref *)(A.data() + stride_a * i), (const int *)&lda_ref,
               (const fp_ref *)(B.data() + stride_b * i), (const int *)&ldb_ref,
               (const fp_ref *)&beta, (fp_ref *)(C_ref.data() + stride_c * i),
               (const int *)&ldc_ref);
    }

    // Call DPC++ GEMM_BATCH_STRIDE.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);

    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<fp, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<fp, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        onemkl::blas::gemm_batch(main_queue, transa, transb, m, n, k, alpha, A_buffer, lda,
                                 stride_a, B_buffer, ldb, stride_b, beta, C_buffer, ldc, stride_c,
                                 batch_size);
#else
        TEST_RUN_CT(main_queue, onemkl::blas::gemm_batch,
                    (main_queue, transa, transb, m, n, k, alpha, A_buffer, lda, stride_a, B_buffer,
                     ldb, stride_b, beta, C_buffer, ldc, stride_c, batch_size));
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GEMM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
#ifdef ENABLE_CUBLAS_BACKEND
        // GEMM_BATCH_STRIDE currently not supported with CUBLAS backend
        std::string error_msg(error.what());
        if (error_msg.compare("Not implemented for cublas") == 0) {
            return true;
        }
#endif
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good;
    {
        auto C_accessor = C_buffer.template get_access<access::mode::read>();
        good            = check_equal_matrix(C_accessor, C_ref, stride_c * batch_size, 1,
                                  stride_c * batch_size, 10 * k, std::cout);
    }

    return good;
}

class GemmBatchStrideTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GemmBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUE(test<float>(GetParam(), 5));
}

TEST_P(GemmBatchStrideTests, RealDoublePrecision) {
    EXPECT_TRUE(test<double>(GetParam(), 5));
}

TEST_P(GemmBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUE(test<std::complex<float>>(GetParam(), 5));
}

TEST_P(GemmBatchStrideTests, ComplexDoublePrecision) {
    EXPECT_TRUE(test<std::complex<double>>(GetParam(), 5));
}

INSTANTIATE_TEST_SUITE_P(GemmBatchStrideTestSuite, GemmBatchStrideTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
