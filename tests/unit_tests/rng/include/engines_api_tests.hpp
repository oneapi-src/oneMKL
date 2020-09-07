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

#ifndef _RNG_ENGINES_API_TESTS_HPP__
#define _RNG_ENGINES_API_TESTS_HPP__

#include <cstdint>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "oneapi/mkl.hpp"

#include "rng_test_common.hpp"

template <typename Engine>
class engines_constructors_test {
public:
    template <typename Queue, typename... Args>
    void operator()(Queue queue, Args... args) {
        // initialize rng objects
        Engine engine1(queue, SEED);
        Engine engine2(queue, args...);
        Engine engine3(engine1);
        Engine engine4 = std::move(Engine(queue, SEED));

        oneapi::mkl::rng::bits<std::uint32_t> distr;

        // prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN);
        std::vector<std::uint32_t> r2(N_GEN);
        std::vector<std::uint32_t> r3(N_GEN);
        std::vector<std::uint32_t> r4(N_GEN);
        {
            cl::sycl::buffer<std::uint32_t, 1> r1_buffer(r1.data(), r1.size());
            cl::sycl::buffer<std::uint32_t, 1> r2_buffer(r2.data(), r2.size());
            cl::sycl::buffer<std::uint32_t, 1> r3_buffer(r3.data(), r3.size());
            cl::sycl::buffer<std::uint32_t, 1> r4_buffer(r4.data(), r4.size());
            try {
                oneapi::mkl::rng::generate(distr, engine1, N_GEN, r1_buffer);
                oneapi::mkl::rng::generate(distr, engine2, N_GEN, r2_buffer);
                oneapi::mkl::rng::generate(distr, engine3, N_GEN, r3_buffer);
                oneapi::mkl::rng::generate(distr, engine4, N_GEN, r4_buffer);
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
        status = (check_equal_vector(r1, r2) && check_equal_vector(r1, r3) &&
                  check_equal_vector(r1, r4));
    }

    int status = test_passed;
};

template <typename Engine>
class engines_copy_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        // initialize rng objects
        Engine engine1(queue, SEED);
        Engine engine2(engine1);

        oneapi::mkl::rng::bits<std::uint32_t> distr;

        // prepare arrays for random numbers
        std::vector<std::uint32_t> r1(N_GEN);
        std::vector<std::uint32_t> r2(N_GEN);
        std::vector<std::uint32_t> r3(N_GEN);
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

        Engine engine3 = engine1;
        Engine engine4 = std::move(engine2);
        {
            cl::sycl::buffer<std::uint32_t, 1> r1_buffer(r1.data(), r1.size());
            cl::sycl::buffer<std::uint32_t, 1> r2_buffer(r2.data(), r2.size());
            cl::sycl::buffer<std::uint32_t, 1> r3_buffer(r3.data(), r3.size());
            try {
                oneapi::mkl::rng::generate(distr, engine1, N_GEN, r1_buffer);
                oneapi::mkl::rng::generate(distr, engine3, N_GEN, r2_buffer);
                oneapi::mkl::rng::generate(distr, engine4, N_GEN, r3_buffer);
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
        status = (check_equal_vector(r1, r2) && check_equal_vector(r1, r3));
    }

    int status = test_passed;
};

#endif // _RNG_ENGINES_API_TESTS_HPP__
