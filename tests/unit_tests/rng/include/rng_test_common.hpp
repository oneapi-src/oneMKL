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

#ifndef _RNG_TEST_COMMON_HPP__
#define _RNG_TEST_COMMON_HPP__

#include <iostream>
#include <limits>

#include "test_helper.hpp"

#define SEED  777
#define N_GEN 1000

// Defines for skip_ahead and leapfrog tests
#define N_ENGINES     6
#define N_PORTION     100
#define N_GEN_SERVICE (N_ENGINES * N_PORTION)

// defines for skip_ahead_ex tests
#define N_SKIP     ((std::uint64_t)pow(2, 62))
#define SKIP_TIMES ((std::int32_t)pow(2, 14))
#define NUM_TO_SKIP \
    { 0, (std::uint64_t)pow(2, 12) }

// Correctness checking.
static inline bool check_equal(float x, float x_ref) {
    float bound = std::numeric_limits<float>::epsilon();
    float aerr = std::abs(x - x_ref);
    return (aerr <= bound);
}

static inline bool check_equal(double x, double x_ref) {
    double bound = std::numeric_limits<double>::epsilon();
    double aerr = std::abs(x - x_ref);
    return (aerr <= bound);
}

static inline bool check_equal(std::uint32_t x, std::uint32_t x_ref) {
    return x == x_ref;
}

static inline bool check_equal(std::uint64_t x, std::uint64_t x_ref) {
    return x == x_ref;
}

template <typename Fp, typename AllocType>
static inline bool check_equal_vector(std::vector<Fp, AllocType>& r1,
                                      std::vector<Fp, AllocType>& r2) {
    for (int i = 0; i < r1.size(); i++) {
        if (!check_equal(r1[i], r2[i])) {
            return false;
            break;
        }
    }
    return true;
}

template <typename Fp, typename AllocType>
static inline bool leapfrog_check(std::vector<Fp, AllocType>& r1, std::vector<Fp, AllocType>& r2,
                                  int n_portion, int n_engines) {
    int j = 0;
    for (int i = 0; i < n_engines; i++) {
        for (int k = 0; k < n_portion / 2; k++) {
            for (int p = 0; p < 2; p++) {
                if (!check_equal(r2[j++], r1[k * n_engines + i * 2 + p])) {
                    return false;
                    break;
                }
            }
        }
    }
    return true;
}

template <typename Test>
class rng_test {
public:
    // method to call any tests, switch between rt and ct
    template <typename... Args>
    int operator()(cl::sycl::device* dev, Args... args) {
        auto exception_handler = [](cl::sycl::exception_list exceptions) {
            for (std::exception_ptr const& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                }
                catch (cl::sycl::exception const& e) {
                    std::cout << "Caught asynchronous SYCL exception during ASUM:\n"
                              << e.what() << std::endl;
                    print_error_code(e);
                }
            }
        };

#ifdef ENABLE_CURAND_BACKEND // w/a for cuda backend hangs when there are several queues with different contexts
        static cl::sycl::device* previous_device = nullptr;
        static cl::sycl::context* context = nullptr;

        if ((previous_device != dev)) {
            previous_device = dev;
            if (context != nullptr) {
                delete context;
            }
            context = new cl::sycl::context(*dev);
        }

        cl::sycl::queue queue(*context, *dev, exception_handler);
#else
        cl::sycl::queue queue(*dev, exception_handler);
#endif

#ifdef CALL_RT_API
        test_(queue, args...);
#else
        TEST_RUN_CT_SELECT(queue, test_, args...);
#endif

        return test_.status;
    }

protected:
    Test test_;
};

#endif // _RNG_TEST_COMMON_HPP__
