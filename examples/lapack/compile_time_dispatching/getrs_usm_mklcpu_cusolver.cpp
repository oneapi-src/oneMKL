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
*       This example demonstrates use of oneapi::mkl::lapack::getrf and
*       oneapi::mkl::lapack::getrs to perform LU factorization and compute
*       the solution on both an Intel CPU device and NVIDIA GPU device.
*
*       This example demonstrates only single precision (float) data type
*       for matrix data
*
*******************************************************************************/

// STL includes
#include <iostream>
#include <vector>

// oneMKL/SYCL includes
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

// local includes
#include "example_helper.hpp"

//
// Main example for LU consisting of initialization of
// a general dense A matrix.
// Then the LU factorization
// A = P * L * U
// is performed followed by solving a system of linear
// equations using the computed LU factorization, with
// multiple right-hand sides.
// Finally the results are post processed.
//

int run_getrs_example(const sycl::device& cpu_device, const sycl::device& gpu_device)
{
    // Matrix sizes and leading dimensions
    std::int64_t m    = 23;
    std::int64_t n    = 23;
    std::int64_t nrhs  = 23;
    std::int64_t lda  = 32;
    std::int64_t ldb  = 32;
    std::int64_t A_size = n * lda;
    std::int64_t B_size = nrhs * ldb;
    std::int64_t ipiv_size = n;

    // Catch asynchronous exceptions for CPU and GPU
    auto cpu_error_handler = [&] (sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(oneapi::mkl::lapack::exception const& e) {
                // Handle LAPACK related exceptions happened during asynchronous call
                std::cerr << "Caught asynchronous LAPACK exception on CPU device during GETRF or GETRS:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
                std::cerr << "\tinfo: " << e.info() << std::endl;
            } catch(sycl::exception const& e) {
                // Handle not LAPACK related exceptions happened during asynchronous call
                std::cerr << "Caught asynchronous SYCL exception on CPU device during GETRF or GETRS:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };
    auto gpu_error_handler = [&] (sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(oneapi::mkl::lapack::exception const& e) {
                // Handle LAPACK related exceptions happened during asynchronous call
                std::cerr << "Caught asynchronous LAPACK exception on CPU device during GETRF or GETRS:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
                std::cerr << "\tinfo: " << e.info() << std::endl;
            } catch(sycl::exception const& e) {
                // Handle not LAPACK related exceptions happened during asynchronous call
                std::cerr << "Caught asynchronous SYCL exception on CPU device during GETRF or GETRS:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    //
    // Data preparation on host
    //
    float *A = (float *)calloc(A_size, sizeof(float));
    float *B = (float *)calloc(B_size, sizeof(float));
    float *result_cpu = (float *)calloc(B_size, sizeof(float));
    float *result_gpu = (float *)calloc(B_size, sizeof(float));
    if (!A || !B || !result_cpu || !result_gpu) {
        throw std::runtime_error("Failed to allocate memory on host.");
    }
    rand_matrix(A, oneapi::mkl::transpose::nontrans, n, n, lda);
    rand_matrix(B, oneapi::mkl::transpose::nontrans, n, nrhs, ldb);


    //
    // Preparation on CPU
    //
    sycl::queue cpu_queue(cpu_device, cpu_error_handler);
    sycl::context cpu_context = cpu_queue.get_context();
    sycl::event cpu_getrf_done;
    sycl::event cpu_getrs_done;

    float *cpu_A = sycl::malloc_device<float>(A_size * sizeof(float), cpu_queue);
    float *cpu_B = sycl::malloc_device<float>(B_size * sizeof(float), cpu_queue);
    std::int64_t *cpu_ipiv = sycl::malloc_device<std::int64_t>(ipiv_size * sizeof(std::int64_t), cpu_queue);

    std::int64_t cpu_getrf_scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<float>(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> {cpu_queue}, m, n, lda);
    std::int64_t cpu_getrs_scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<float>(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> {cpu_queue}, oneapi::mkl::transpose::nontrans, n, nrhs, lda, ldb);
    float *cpu_getrf_scratchpad = sycl::malloc_device<float>(cpu_getrf_scratchpad_size * sizeof(float), cpu_device, cpu_context);
    float *cpu_getrs_scratchpad = sycl::malloc_device<float>(cpu_getrs_scratchpad_size * sizeof(float), cpu_device, cpu_context);
    if (!cpu_A || !cpu_B || !cpu_ipiv || !cpu_getrf_scratchpad || !cpu_getrs_scratchpad) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    // copy data from host to CPU device
    cpu_queue.memcpy(cpu_A, A, A_size * sizeof(float)).wait();
    cpu_queue.memcpy(cpu_B, B, B_size * sizeof(float)).wait();


    //
    // Preparation on GPU
    //
    sycl::queue gpu_queue(gpu_device, gpu_error_handler);
    sycl::context gpu_context = gpu_queue.get_context();
    sycl::event gpu_getrf_done;
    sycl::event gpu_getrs_done;

    float *gpu_A = sycl::malloc_device<float>(A_size * sizeof(float), gpu_queue);
    float *gpu_B = sycl::malloc_device<float>(B_size * sizeof(float), gpu_queue);
    std::int64_t *gpu_ipiv = sycl::malloc_device<std::int64_t>(ipiv_size * sizeof(std::int64_t), gpu_queue);

    std::int64_t gpu_getrf_scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<float>(oneapi::mkl::backend_selector<oneapi::mkl::backend::cusolver> {gpu_queue}, m, n, lda);
    std::int64_t gpu_getrs_scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<float>(oneapi::mkl::backend_selector<oneapi::mkl::backend::cusolver> {gpu_queue}, oneapi::mkl::transpose::nontrans, n, nrhs, lda, ldb);
    float *gpu_getrf_scratchpad = sycl::malloc_device<float>(gpu_getrf_scratchpad_size * sizeof(float), gpu_device, gpu_context);
    float *gpu_getrs_scratchpad = sycl::malloc_device<float>(gpu_getrs_scratchpad_size * sizeof(float), gpu_device, gpu_context);
    if (!gpu_A || !gpu_B || !gpu_ipiv || !gpu_getrf_scratchpad) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    // copy data from host to CPU device
    gpu_queue.memcpy(gpu_A, A, A_size * sizeof(float)).wait();
    gpu_queue.memcpy(gpu_B, B, B_size * sizeof(float)).wait();


    //
    // Execute on CPU and GPU devices
    //

    cpu_getrf_done = oneapi::mkl::lapack::getrf(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> {cpu_queue}, m, n, cpu_A, lda, cpu_ipiv, cpu_getrf_scratchpad, cpu_getrf_scratchpad_size);
    cpu_getrs_done = oneapi::mkl::lapack::getrs(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> {cpu_queue}, oneapi::mkl::transpose::nontrans, n, nrhs, cpu_A, lda, cpu_ipiv, cpu_B, ldb, cpu_getrs_scratchpad, cpu_getrs_scratchpad_size, {cpu_getrf_done});
    gpu_getrf_done = oneapi::mkl::lapack::getrf(oneapi::mkl::backend_selector<oneapi::mkl::backend::cusolver> {gpu_queue}, m, n, gpu_A, lda, gpu_ipiv, gpu_getrf_scratchpad, gpu_getrf_scratchpad_size);
    gpu_getrs_done = oneapi::mkl::lapack::getrs(oneapi::mkl::backend_selector<oneapi::mkl::backend::cusolver> {gpu_queue}, oneapi::mkl::transpose::nontrans, n, nrhs, gpu_A, lda, gpu_ipiv, gpu_B, ldb, gpu_getrs_scratchpad, gpu_getrs_scratchpad_size, {cpu_getrf_done});


    // Wait until calculations are done
    cpu_queue.wait_and_throw();
    gpu_queue.wait_and_throw();


    //
    // Post Processing
    //
    // copy data from CPU device back to host
    cpu_queue.memcpy(result_cpu, cpu_B, B_size * sizeof(float)).wait();

    // copy data from GPU device back to host
    gpu_queue.memcpy(result_gpu, gpu_B, B_size * sizeof(float)).wait();

    // compare results from CPU and GPU devices
    int ret = check_equal_matrix(result_cpu, result_gpu, n, nrhs, ldb);

    sycl::free(cpu_A, cpu_queue);
    sycl::free(cpu_B, cpu_queue);
    sycl::free(cpu_ipiv, cpu_queue);
    sycl::free(cpu_getrf_scratchpad, cpu_queue);
    sycl::free(cpu_getrs_scratchpad, cpu_queue);

    sycl::free(gpu_A, gpu_queue);
    sycl::free(gpu_B, gpu_queue);
    sycl::free(gpu_ipiv, gpu_queue);
    sycl::free(gpu_getrf_scratchpad, gpu_queue);
    sycl::free(gpu_getrs_scratchpad, gpu_queue);

    free(A);
    free(B);
    free(result_cpu);
    free(result_gpu);

    return ret;
}


//
// Description of example setup, apis used and supported floating point type precisions
//

void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# LU Factorization and Solve Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Computes LU Factorization A = P * L * U" << std::endl;
    std::cout << "# and uses it to solve for X in a system of linear equations:" << std::endl;
    std::cout << "#   AX = B" << std::endl;
    std::cout << "# where A is a general dense matrix and B is a matrix whose columns" << std::endl;
    std::cout << "# are the right-hand sides for the systems of equations." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   getrf and getrs" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Running on both Intel CPU and NVIDIA GPU devices" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;

}


//
// Main entry point for example.
//
int main(int argc, char **argv) {
    print_example_banner();

    try{
        sycl::device cpu_dev((sycl::cpu_selector()));
        sycl::device gpu_dev((sycl::gpu_selector()));

        unsigned int vendor_id = gpu_dev.get_info<sycl::info::device::vendor_id>();
        if (vendor_id != NVIDIA_ID) {
            std::cerr << "FAILED: NVIDIA GPU device not found." << std::endl;
            return 1;
        }
        std::cout << "Running LAPACK GETRS USM example" << std::endl;
        std::cout << "Running with single precision real data type on:" << std::endl;
        std::cout << "\tCPU device :" << cpu_dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "\tGPU device :" << gpu_dev.get_info<sycl::info::device::name>() << std::endl;

        int ret = run_getrs_example(cpu_dev, gpu_dev);
        if (ret) {
            std::cout << "LAPACK GETRS USM example ran OK: CPU and GPU results match" << std::endl;
        } else {
            std::cerr << "LAPACK GETRS USM example FAILED: CPU and GPU results do not match" << std::endl;
        }

    } catch(oneapi::mkl::lapack::exception const& e) {
        // Handle LAPACK related exceptions happened during synchronous call
        std::cerr << "Caught synchronous LAPACK exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tinfo: " << e.info() << std::endl;
        return 1;
    } catch(sycl::exception const& e) {
        // Handle not LAPACK related exceptions happened during synchronous call
        std::cerr << "Caught synchronous SYCL exception:"  << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    } catch(std::exception const& e) {
        // Handle not SYCL related exceptions happened during synchronous call
        std::cerr << "Caught synchronous std::exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}
