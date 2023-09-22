/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef _RNG_DEVICE_SKIP_AHEAD_TEST_HPP__
#define _RNG_DEVICE_SKIP_AHEAD_TEST_HPP__

#include <cstdint>
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/rng/device.hpp"

#include "rng_device_test_common.hpp"

template <typename Engine>
class skip_ahead_test {
public:
    template <typename Queue>
    void operator()(Queue queue) {
        std::vector<std::uint32_t> r(N_GEN);
        std::vector<std::uint32_t> r_ref(N_GEN);

        try {
            sycl::range<1> range(N_GEN / Engine::vec_size);

            sycl::buffer<std::uint32_t, 1> buf(r);
            auto event = queue.submit([&](sycl::handler& cgh) {
                auto acc = buf.template get_access<sycl::access::mode::write>(cgh);
                cgh.parallel_for(range, [=](sycl::item<1> item) {
                    size_t id = item.get_id(0);
                    Engine engine(SEED);
                    oneapi::mkl::rng::device::skip_ahead(engine, id * Engine::vec_size);
                    oneapi::mkl::rng::device::bits<> distr;
                    auto res = oneapi::mkl::rng::device::generate(distr, engine);
                    if constexpr(Engine::vec_size == 1) {
                        acc[id] = res;
                    }
                    else {
                        res.store(id, acc.get_pointer());
                    }
                });
            });
            event.wait_and_throw();
        }
        catch (const oneapi::mkl::unimplemented& e) {
            status = test_skipped;
            return;
        }
        catch(sycl::exception const& e) {
            std::cout << "SYCL exception during generation" << std::endl
                        << e.what() << std::endl << "Error code: " << get_error_code(e) << std::endl;
            status = test_failed;
            return;
        }

        // validation
        Engine engine(SEED);
        oneapi::mkl::rng::device::bits<> distr;
        for(int i = 0; i < N_GEN; i++) {
            if constexpr(Engine::vec_size == 1) {
                r_ref[i] = oneapi::mkl::rng::device::generate(distr, engine);
            }
            else {
                r_ref[i] =  oneapi::mkl::rng::device::generate_single(distr, engine);
            }
        }

        status = check_equal_vector_device(r, r_ref);
    }

    int status = test_passed;
};

#endif // _RNG_DEVICE_SKIP_AHEAD_TEST_HPP__
