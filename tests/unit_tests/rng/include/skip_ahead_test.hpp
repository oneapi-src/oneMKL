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

#ifndef _RNG_TEST_SKIP_AHEAD_TEST_HPP__
#define _RNG_TEST_SKIP_AHEAD_TEST_HPP__

#include <cstdint>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "oneapi/mkl.hpp"

#include "rng_test_common.hpp"

template <typename Engine>
class skip_ahead_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        // initialize rng objects
        Engine engine(queue);

        std::vector<Engine*> engines;

        oneapi::mkl::rng::bits<std::uint32_t> distr;

        // prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN_SERVICE);
        std::vector<std::uint32_t> r2(N_GEN_SERVICE);

        // perform skip
        for (int i = 0; i < N_ENGINES; i++) {
            engines.push_back(new Engine(queue));
            oneapi::mkl::rng::skip_ahead(*(engines[i]), i * N_PORTION);
        }

        {
            cl::sycl::buffer<std::uint32_t, 1> r_buffer(r1.data(), r1.size());
            std::vector<cl::sycl::buffer<std::uint32_t, 1>> r_buffers;
            for (int i = 0; i < N_ENGINES; i++) {
                r_buffers.push_back(
                    cl::sycl::buffer<std::uint32_t, 1>(r2.data() + i * N_PORTION, N_PORTION));
            }
            try {
                oneapi::mkl::rng::generate(distr, engine, N_GEN_SERVICE, r_buffer);
                for (int i = 0; i < N_ENGINES; i++) {
                    oneapi::mkl::rng::generate(distr, *(engines[i]), N_PORTION, r_buffers[i]);
                }
            }
            catch (cl::sycl::exception const& e) {
                std::cout << "SYCL exception during generation" << std::endl
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
                status = test_failed;
                return;
            }
        } // buffers life-time ends

        // clear memory
        for (int i = 0; i < N_ENGINES; i++) {
            delete engines[i];
        }

        // validation
        status = check_equal_vector(r1, r2);
    }

    int status = test_passed;
};

template <typename Engine>
class skip_ahead_ex_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        Engine engine1(queue);
        Engine engine2(queue);

        oneapi::mkl::rng::bits<std::uint32_t> distr;

        // prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN);
        std::vector<std::uint32_t> r2(N_GEN);

        // perform skip

        for (int j = 0; j < SKIP_TIMES; j++) {
            oneapi::mkl::rng::skip_ahead(engine1, N_SKIP);
        }
        oneapi::mkl::rng::skip_ahead(engine2, NUM_TO_SKIP);

        {
            cl::sycl::buffer<std::uint32_t, 1> r1_buffer(r1.data(), r1.size());
            cl::sycl::buffer<std::uint32_t, 1> r2_buffer(r2.data(), r2.size());

            try {
                oneapi::mkl::rng::generate(distr, engine1, N_GEN, r1_buffer);
                oneapi::mkl::rng::generate(distr, engine2, N_GEN, r2_buffer);
            }
            catch (cl::sycl::exception const& e) {
                std::cout << "SYCL exception during generation" << std::endl
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
                status = test_failed;
                return;
            }
        } // buffers life-time ends

        // validation
        status = check_equal_vector(r1, r2);
    }

    int status = test_passed;
};

#endif // _RNG_TEST_SKIP_AHEAD_TEST_HPP__
