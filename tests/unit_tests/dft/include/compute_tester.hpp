/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef ONEMKL_COMPUTE_TESTER_HPP
#define ONEMKL_COMPUTE_TESTER_HPP

#include <algorithm>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"
#include "test_helper.hpp"
#include "test_common.hpp"
#include "reference_dft.hpp"

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
struct DFT_Test {
    using descriptor_t = oneapi::mkl::dft::descriptor<precision, domain>;

    template <typename ElemT>
    using usm_allocator_t = sycl::usm_allocator<ElemT, sycl::usm::alloc::shared, 64>;

    using PrecisionType =
        typename std::conditional_t<precision == oneapi::mkl::dft::precision::SINGLE, float,
                                    double>;

    using FwdInputType = typename std::conditional_t<domain == oneapi::mkl::dft::domain::REAL,
                                                     PrecisionType, std::complex<PrecisionType>>;
    using FwdOutputType = std::complex<PrecisionType>;

    enum class MemoryAccessModel { buffer, usm };

    const std::int64_t size;
    const std::int64_t conjugate_even_size;
    double abs_error_margin;
    double rel_error_margin;

    sycl::device *dev;
    sycl::queue sycl_queue;
    sycl::context cxt;

    std::vector<FwdInputType> input;
    std::vector<PrecisionType> input_re;
    std::vector<PrecisionType> input_im;
    std::vector<FwdOutputType> out_host_ref;

    DFT_Test(sycl::device *dev, std::int64_t size)
            : size{ static_cast<std::int64_t>(size) },
              conjugate_even_size{ 2 * (size / 2 + 1) },
              abs_error_margin{0},
              rel_error_margin{0},
              dev{ dev },
              sycl_queue{ *dev, exception_handler },
              cxt{ sycl_queue.get_context() } {
        input = std::vector<FwdInputType>(size);
        input_re = std::vector<PrecisionType>(size);
        input_im = std::vector<PrecisionType>(size);

        // out_host_ref contains redundant information for domain::REAL
        // tests. This simplifies the test implementation, but increases
        // storage and computational requirements. There is scope for
        // improvement here if test performance becomes an issue.
        out_host_ref = std::vector<FwdOutputType>(size);
        rand_vector(input, size);

        if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
            for (int i = 0; i < input.size(); ++i) {
                input_re[i] = { input[i] };
                input_im[i] = 0;
            }
        }
        else {
            for (int i = 0; i < input.size(); ++i) {
                input_re[i] = { input[i].real() };
                input_im[i] = { input[i].imag() };
            }
        }
    }

    bool skip_test(MemoryAccessModel type) {
        if constexpr (precision == oneapi::mkl::dft::precision::DOUBLE) {
            if (!sycl_queue.get_device().has(sycl::aspect::fp64)) {
                std::cout << "Device does not support double precision." << std::endl;
                return true;
            }
        }
        if (type == MemoryAccessModel::usm &&
            !sycl_queue.get_device().has(sycl::aspect::usm_shared_allocations)) {
            std::cout << "Device does not support usm shared allocations." << std::endl;
            return true;
        }
        return false;
    }

    bool init(MemoryAccessModel type) {
        reference_forward_dft<FwdInputType, FwdOutputType>(input, out_host_ref);
        auto max_norm_ref = *std::max_element(std::begin(out_host_ref), std::end(out_host_ref),
          [](const FwdOutputType& a, const FwdOutputType& b) { return std::abs(a) < std::abs(b); });
        // Heuristic for the average-case error margins
        abs_error_margin = std::abs(max_norm_ref) * std::log2((double)size);
        rel_error_margin = 5.0 * std::log2((double)size);
        return !skip_test(type);
    }

    int test_in_place_buffer();
    int test_in_place_real_real_buffer();
    int test_out_of_place_buffer();
    int test_out_of_place_real_real_buffer();
    int test_in_place_USM();
    int test_in_place_real_real_USM();
    int test_out_of_place_USM();
    int test_out_of_place_real_real_USM();
};

#endif //ONEMKL_COMPUTE_TESTER_HPP
