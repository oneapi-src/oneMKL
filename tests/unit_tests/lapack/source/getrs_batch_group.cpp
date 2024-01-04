/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <complex>
#include <list>
#include <numeric>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl.hpp"
#include "lapack_common.hpp"
#include "lapack_test_controller.hpp"
#include "lapack_accuracy_checks.hpp"
#include "lapack_reference_wrappers.hpp"
#include "test_helper.hpp"

namespace {

const char* accuracy_input = R"(
27182
)";

template <typename fp>
bool accuracy(const sycl::device& dev, uint64_t seed) {
    using fp_real = typename complex_info<fp>::real_type;

    /* Test Parameters */
    std::vector<oneapi::mkl::transpose> trans_vec = { oneapi::mkl::transpose::nontrans,
                                                      oneapi::mkl::transpose::trans };
    std::vector<int64_t> n_vec = { 4, 5 };
    std::vector<int64_t> nrhs_vec = { 9, 6 };
    std::vector<int64_t> lda_vec = { 6, 6 };
    std::vector<int64_t> ldb_vec = { 9, 9 };
    std::vector<int64_t> group_sizes_vec = { 2, 2 };

    int64_t group_count = group_sizes_vec.size();
    int64_t batch_size = std::accumulate(group_sizes_vec.begin(), group_sizes_vec.end(), 0);

    std::list<std::vector<fp>> A_initial_list;
    std::list<std::vector<fp>> B_initial_list;
    std::list<std::vector<fp>> A_list;
    std::list<std::vector<fp>> B_list;
    std::list<std::vector<int64_t>> ipiv_list;

    for (int64_t group_id = 0; group_id < group_count; group_id++) {
        auto trans = trans_vec[group_id];
        auto n = n_vec[group_id];
        auto nrhs = nrhs_vec[group_id];
        auto lda = lda_vec[group_id];
        auto ldb = ldb_vec[group_id];
        auto group_size = group_sizes_vec[group_id];

        /* Allocate and Initialize on host */
        for (int64_t local_id = 0; local_id < group_size; local_id++) {
            A_initial_list.emplace_back(lda * n);
            auto& A_initial = A_initial_list.back();
            rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A_initial, lda);

            A_list.emplace_back(A_initial);
            auto& A = A_list.back();

            B_initial_list.emplace_back(ldb * nrhs);
            auto& B_initial = B_initial_list.back();
            rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B_initial, lda);

            B_list.emplace_back(B_initial);
            auto& B = B_list.back();

            ipiv_list.emplace_back(n);
            auto& ipiv = ipiv_list.back();

            auto info = reference::getrf(n, n, A.data(), lda, ipiv.data());
            if (info != 0) {
                test_log::lout << "Reference getrf failed with info = " << info << std::endl;
                return false;
            }
        }
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> A_dev_list;
        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> B_dev_list;
        std::list<std::vector<int64_t, sycl::usm_allocator<int64_t, sycl::usm::alloc::shared>>>
            ipiv_dev_list;
        fp** A_dev_ptrs = sycl::malloc_shared<fp*>(batch_size, queue);
        fp** B_dev_ptrs = sycl::malloc_shared<fp*>(batch_size, queue);
        int64_t** ipiv_dev_ptrs = sycl::malloc_shared<int64_t*>(batch_size, queue);

        /* Allocate on device */
        sycl::usm_allocator<fp, sycl::usm::alloc::shared> usm_fp_allocator{ queue.get_context(),
                                                                            dev };
        sycl::usm_allocator<int64_t, sycl::usm::alloc::shared> usm_int_allocator{
            queue.get_context(), dev
        };
        auto A_iter = A_list.begin();
        auto B_iter = B_list.begin();
        auto ipiv_iter = ipiv_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_iter++, B_iter++, ipiv_iter++) {
            A_dev_list.emplace_back(A_iter->size(), usm_fp_allocator);
            B_dev_list.emplace_back(B_iter->size(), usm_fp_allocator);
            ipiv_dev_list.emplace_back(ipiv_iter->size(), usm_int_allocator);
        }

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<fp>(
            queue, trans_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(), ldb_vec.data(),
            group_count, group_sizes_vec.data());
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<fp>,
            trans_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(), ldb_vec.data(),
            group_count, group_sizes_vec.data());
#endif
        auto scratchpad_dev = device_alloc<fp>(queue, scratchpad_size);

        auto A_dev_iter = A_dev_list.begin();
        auto B_dev_iter = B_dev_list.begin();
        auto ipiv_dev_iter = ipiv_dev_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_dev_iter++, B_dev_iter++, ipiv_dev_iter++) {
            A_dev_ptrs[global_id] = A_dev_iter->data();
            B_dev_ptrs[global_id] = B_dev_iter->data();
            ipiv_dev_ptrs[global_id] = ipiv_dev_iter->data();
        }

        A_iter = A_list.begin();
        B_iter = B_list.begin();
        ipiv_iter = ipiv_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_iter++, B_iter++, ipiv_iter++) {
            host_to_device_copy(queue, A_iter->data(), A_dev_ptrs[global_id], A_iter->size());
            host_to_device_copy(queue, B_iter->data(), B_dev_ptrs[global_id], B_iter->size());
            host_to_device_copy(queue, ipiv_iter->data(), ipiv_dev_ptrs[global_id],
                                ipiv_iter->size());
        }
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::getrs_batch(queue, trans_vec.data(), n_vec.data(), nrhs_vec.data(),
                                         A_dev_ptrs, lda_vec.data(), ipiv_dev_ptrs, B_dev_ptrs,
                                         ldb_vec.data(), group_count, group_sizes_vec.data(),
                                         scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::mkl::lapack::getrs_batch, trans_vec.data(),
                                  n_vec.data(), nrhs_vec.data(), A_dev_ptrs, lda_vec.data(),
                                  ipiv_dev_ptrs, B_dev_ptrs, ldb_vec.data(), group_count,
                                  group_sizes_vec.data(), scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        B_iter = B_list.begin();
        for (int64_t global_id = 0; global_id < batch_size; global_id++, B_iter++) {
            device_to_host_copy(queue, B_dev_ptrs[global_id], B_iter->data(), B_iter->size());
        }
        queue.wait_and_throw();
        if (scratchpad_dev) {
            sycl::free(scratchpad_dev, queue);
        }
        if (A_dev_ptrs) {
            sycl::free(A_dev_ptrs, queue);
        }
        if (B_dev_ptrs) {
            sycl::free(B_dev_ptrs, queue);
        }
        if (ipiv_dev_ptrs) {
            sycl::free(ipiv_dev_ptrs, queue);
        }
    }

    bool result = true;

    int64_t global_id = 0;
    auto A_iter = A_list.begin();
    auto B_iter = B_list.begin();
    auto ipiv_iter = ipiv_list.begin();
    auto A_initial_iter = A_initial_list.begin();
    auto B_initial_iter = B_initial_list.begin();
    for (int64_t group_id = 0; group_id < group_count; group_id++) {
        auto trans = trans_vec[group_id];
        auto n = n_vec[group_id];
        auto nrhs = nrhs_vec[group_id];
        auto lda = lda_vec[group_id];
        auto ldb = ldb_vec[group_id];
        auto group_size = group_sizes_vec[group_id];
        for (int64_t local_id = 0; local_id < group_size; local_id++, global_id++, A_iter++,
                     B_iter++, ipiv_iter++, A_initial_iter++, B_initial_iter++) {
            if (!check_getrs_accuracy(trans, n, nrhs, *B_iter, ldb, *A_initial_iter, lda,
                                      *B_initial_iter)) {
                test_log::lout << "batch routine (" << global_id << ", " << group_id << ", "
                               << local_id << ") (global_id, group_id, local_id) failed"
                               << std::endl;
                result = false;
            }
        }
    }

    return result;
}

const char* dependency_input = R"(
1
)";

template <typename fp>
bool usm_dependency(const sycl::device& dev, uint64_t seed) {
    using fp_real = typename complex_info<fp>::real_type;

    /* Test Parameters */
    std::vector<oneapi::mkl::transpose> trans_vec = { oneapi::mkl::transpose::nontrans };
    std::vector<int64_t> n_vec = { 1 };
    std::vector<int64_t> nrhs_vec = { 1 };
    std::vector<int64_t> lda_vec = { 1 };
    std::vector<int64_t> ldb_vec = { 1 };
    std::vector<int64_t> group_sizes_vec = { 1 };

    int64_t group_count = group_sizes_vec.size();
    int64_t batch_size = std::accumulate(group_sizes_vec.begin(), group_sizes_vec.end(), 0);

    std::list<std::vector<fp>> A_initial_list;
    std::list<std::vector<fp>> B_initial_list;
    std::list<std::vector<fp>> A_list;
    std::list<std::vector<fp>> B_list;
    std::list<std::vector<int64_t>> ipiv_list;

    for (int64_t group_id = 0; group_id < group_count; group_id++) {
        auto trans = trans_vec[group_id];
        auto n = n_vec[group_id];
        auto nrhs = nrhs_vec[group_id];
        auto lda = lda_vec[group_id];
        auto ldb = ldb_vec[group_id];
        auto group_size = group_sizes_vec[group_id];

        /* Allocate and Initialize on host */
        for (int64_t local_id = 0; local_id < group_size; local_id++) {
            A_initial_list.emplace_back(lda * n);
            auto& A_initial = A_initial_list.back();
            rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, n, A_initial, lda);

            A_list.emplace_back(A_initial);
            auto& A = A_list.back();

            B_initial_list.emplace_back(ldb * nrhs);
            auto& B_initial = B_initial_list.back();
            rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B_initial, lda);

            B_list.emplace_back(B_initial);
            auto& B = B_list.back();

            ipiv_list.emplace_back(n);
            auto& ipiv = ipiv_list.back();

            auto info = reference::getrf(n, n, A.data(), lda, ipiv.data());
            if (info != 0) {
                test_log::lout << "Reference getrf failed with info = " << info << std::endl;
                return false;
            }
        }
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> A_dev_list;
        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> B_dev_list;
        std::list<std::vector<int64_t, sycl::usm_allocator<int64_t, sycl::usm::alloc::shared>>>
            ipiv_dev_list;
        fp** A_dev_ptrs = sycl::malloc_shared<fp*>(batch_size, queue);
        fp** B_dev_ptrs = sycl::malloc_shared<fp*>(batch_size, queue);
        int64_t** ipiv_dev_ptrs = sycl::malloc_shared<int64_t*>(batch_size, queue);

        /* Allocate on device */
        sycl::usm_allocator<fp, sycl::usm::alloc::shared> usm_fp_allocator{ queue.get_context(),
                                                                            dev };
        sycl::usm_allocator<int64_t, sycl::usm::alloc::shared> usm_int_allocator{
            queue.get_context(), dev
        };
        auto A_iter = A_list.begin();
        auto B_iter = B_list.begin();
        auto ipiv_iter = ipiv_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_iter++, B_iter++, ipiv_iter++) {
            A_dev_list.emplace_back(A_iter->size(), usm_fp_allocator);
            B_dev_list.emplace_back(B_iter->size(), usm_fp_allocator);
            ipiv_dev_list.emplace_back(ipiv_iter->size(), usm_int_allocator);
        }

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<fp>(
            queue, trans_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(), ldb_vec.data(),
            group_count, group_sizes_vec.data());
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<fp>,
            trans_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(), ldb_vec.data(),
            group_count, group_sizes_vec.data());
#endif
        auto scratchpad_dev = device_alloc<fp>(queue, scratchpad_size);

        auto A_dev_iter = A_dev_list.begin();
        auto B_dev_iter = B_dev_list.begin();
        auto ipiv_dev_iter = ipiv_dev_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_dev_iter++, B_dev_iter++, ipiv_dev_iter++) {
            A_dev_ptrs[global_id] = A_dev_iter->data();
            B_dev_ptrs[global_id] = B_dev_iter->data();
            ipiv_dev_ptrs[global_id] = ipiv_dev_iter->data();
        }

        A_iter = A_list.begin();
        B_iter = B_list.begin();
        ipiv_iter = ipiv_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_iter++, B_iter++, ipiv_iter++) {
            host_to_device_copy(queue, A_iter->data(), A_dev_ptrs[global_id], A_iter->size());
            host_to_device_copy(queue, B_iter->data(), B_dev_ptrs[global_id], B_iter->size());
            host_to_device_copy(queue, ipiv_iter->data(), ipiv_dev_ptrs[global_id],
                                ipiv_iter->size());
        }
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::getrs_batch(
            queue, trans_vec.data(), n_vec.data(), nrhs_vec.data(), A_dev_ptrs, lda_vec.data(),
            ipiv_dev_ptrs, B_dev_ptrs, ldb_vec.data(), group_count, group_sizes_vec.data(),
            scratchpad_dev, scratchpad_size, std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::mkl::lapack::getrs_batch,
                                  trans_vec.data(), n_vec.data(), nrhs_vec.data(), A_dev_ptrs,
                                  lda_vec.data(), ipiv_dev_ptrs, B_dev_ptrs, ldb_vec.data(),
                                  group_count, group_sizes_vec.data(), scratchpad_dev,
                                  scratchpad_size, std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        if (scratchpad_dev) {
            sycl::free(scratchpad_dev, queue);
        }
        if (A_dev_ptrs) {
            sycl::free(A_dev_ptrs, queue);
        }
        if (B_dev_ptrs) {
            sycl::free(B_dev_ptrs, queue);
        }
        if (ipiv_dev_ptrs) {
            sycl::free(ipiv_dev_ptrs, queue);
        }
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY_USM(GetrsBatchGroup);
INSTANTIATE_GTEST_SUITE_DEPENDENCY(GetrsBatchGroup);
