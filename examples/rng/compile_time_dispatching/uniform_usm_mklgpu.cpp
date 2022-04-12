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

/*
*
*  Content:
*       This example demonstrates use of DPC++ API oneapi::mkl::rng::uniform distribution with
*       oneapi::mkl::rng::philox4x32x10 random number generator to produce random numbers on
*       an Intel GPU SYCL device with Unified Shared Memory(USM) API.
*
*       The supported data types for random numbers are:
*           float
*
*******************************************************************************/


// stl includes
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "rng_example_helper.hpp"

//
// Main example for Uniform random number generation consisting of
// initialization of random number engine philox4x32x10 object, distribution
// object. Then random number generation performed and
// the output is post-processed and validated.
//
bool run_uniform_example(const sycl::device &dev) {

    //
    // Initialization
    //
    // example parameters defines
    constexpr std::uint64_t seed = 777;
    constexpr std::size_t n = 1000;
    constexpr std::size_t n_print = 10;
    constexpr std::size_t alignment = 64;

    sycl::queue queue(dev, exception_handler);

    // set scalar Type values
    float a(0.0);
    float b(10.0);

    oneapi::mkl::rng::default_engine engine(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu> {queue}, seed);

    oneapi::mkl::rng::uniform<float> distribution(a, b);

    // prepare array for random numbers
    sycl::usm_allocator<float, sycl::usm::alloc::shared, alignment> allocator(queue);
    std::vector<float, decltype(allocator)> r(n, allocator);

    //
    // Perform generation
    //
    sycl::event event_out{};
    try {
        event_out = oneapi::mkl::rng::generate(distribution, engine, n, r.data());
    }
    catch(sycl::exception const& e) {
        std::cout << "\t\tSYCL exception during generation\n"
                  << e.what() << std::endl << "Error code: " << get_error_code(e) << std::endl;
        return false;
    }
    event_out.wait_and_throw();

    //
    // Post Processing
    //

    std::cout << "\n\t\tgeneration parameters:\n";
    std::cout << "\t\t\tseed = " << seed << ", a = " << a << ", b = " << b << std::endl;

    std::cout << "\n\t\tOutput of generator:" << std::endl;

    std::cout << "first "<< n_print << " numbers of " << n << ": " << std::endl;
    for(int i = 0 ; i < n_print; i++) {
        std::cout << r[i] << " ";
    }
    std::cout << std::endl;

    //
    // Validation
    //
    return check_statistics(r, distribution);
}


//
// Description of example setup, APIs used and supported floating point type precisions
//
void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# Generate uniformly distributed random numbers with philox4x32x10\n# generator example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using APIs:" << std::endl;
    std::cout << "#   default_engine uniform" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported precisions:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;

}


//
// Main entry point for example.
//

int main (int argc, char ** argv) {
    print_example_banner();

    sycl::device my_dev = sycl::device(sycl::gpu_selector());

    unsigned int vendor_id = static_cast<unsigned int>(my_dev.get_info<sycl::info::device::vendor_id>());
    if (my_dev.is_gpu() && vendor_id == INTEL_ID) {
        std::cout << "Running RNG uniform usm example on GPU device. Device name is: " << my_dev.get_info<sycl::info::device::name>() << ".\n";
    } else {
        std::cout << "FAILED: INTEL GPU device not found.\n";
        return 1;
    }

    bool is_level0 = my_dev.get_info<sycl::info::device::opencl_c_version>().empty();
    if (is_level0) std::cout << "DPC++ running with Level0 backend.\n";
    else std::cout << "DPC++ running with OpenCL backend.\n";

    std::cout << "\tRunning with single precision real data type:" << std::endl;
    if(!run_uniform_example(my_dev)) {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
