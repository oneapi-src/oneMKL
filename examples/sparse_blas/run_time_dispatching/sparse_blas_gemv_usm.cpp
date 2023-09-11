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

/*
*
*  Content:
*       This example demonstrates use of DPCPP API oneapi::mkl::sparse::gemv
*       using unified shared memory to perform general sparse matrix-vector
*       multiplication on a SYCL device (HOST, CPU, GPU) that is selected
*       during runtime.
*
*       y = alpha * op(A) * x + beta * y
*
*       where op() is defined by one of
*
*           oneapi::mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       This example demonstrates only single precision (float) data type for
*       gemv matrix data
*
*
*******************************************************************************/

// stl includes
#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"

#include "example_helper.hpp"

//
// Main example for Sparse Matrix-Vector Multiply consisting of
// initialization of A matrix, x and y vectors as well as
// scalars alpha and beta.  Then the product
//
// y = alpha * op(A) * x + beta * y
//
// is performed and finally the results are post processed.
//
int run_sparse_matrix_vector_multiply_example(const sycl::device &dev) {
    using fp = float;
    using intType = std::int32_t;

    // Matrix data size
    intType size = 4;
    intType nrows = size * size * size;

    // Set scalar fp values
    fp alpha = set_fp_value(fp(1.0));
    fp beta = set_fp_value(fp(0.0));

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL "
                             "exception during sparse::gemv:\n"
                          << e.what() << std::endl;
            }
        }
    };

    // create execution queue and buffers of matrix data
    sycl::queue main_queue(dev, exception_handler);

    intType *ia, *ja;
    fp *a, *x, *y, *z;
    std::size_t sizea = static_cast<std::size_t>(27 * nrows);
    std::size_t sizeja = static_cast<std::size_t>(27 * nrows);
    std::size_t sizeia = static_cast<std::size_t>(nrows + 1);
    std::size_t sizevec = static_cast<std::size_t>(nrows);

    ia = (intType *)sycl::malloc_shared(sizeia * sizeof(intType), main_queue);
    ja = (intType *)sycl::malloc_shared(sizeja * sizeof(intType), main_queue);
    a = (fp *)sycl::malloc_shared(sizea * sizeof(fp), main_queue);
    x = (fp *)sycl::malloc_shared(sizevec * sizeof(fp), main_queue);
    y = (fp *)sycl::malloc_shared(sizevec * sizeof(fp), main_queue);
    z = (fp *)sycl::malloc_shared(sizevec * sizeof(fp), main_queue);

    if (!ia || !ja || !a || !x || !y || !z) {
        throw std::runtime_error("Failed to allocate USM memory");
    }

    generate_sparse_matrix<fp, intType>(size, ia, ja, a);

    // Init vectors x and y
    for (int i = 0; i < nrows; i++) {
        x[i] = set_fp_value(fp(1.0));
        y[i] = set_fp_value(fp(0.0));
        z[i] = set_fp_value(fp(0.0));
    }

    std::vector<intType *> int_ptr_vec;
    int_ptr_vec.push_back(ia);
    int_ptr_vec.push_back(ja);
    std::vector<fp *> fp_ptr_vec;
    fp_ptr_vec.push_back(a);
    fp_ptr_vec.push_back(x);
    fp_ptr_vec.push_back(y);
    fp_ptr_vec.push_back(z);

    //
    // Execute Matrix Multiply
    //

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
    std::cout << "\n\t\tsparse::gemv parameters:\n";
    std::cout << "\t\t\ttransA = "
              << (transA == oneapi::mkl::transpose::nontrans
                      ? "nontrans"
                      : (transA == oneapi::mkl::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tnrows = " << nrows << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;

    // create and initialize handle for a Sparse Matrix in CSR format
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;

    oneapi::mkl::sparse::init_matrix_handle(main_queue, &handle);

    auto ev_set = oneapi::mkl::sparse::set_csr_data(main_queue, handle, nrows, nrows,
                                                    oneapi::mkl::index_base::zero, ia, ja, a);

    auto ev_opt = oneapi::mkl::sparse::optimize_gemv(main_queue, transA, handle, { ev_set });

    auto ev_gemv =
        oneapi::mkl::sparse::gemv(main_queue, transA, alpha, handle, x, beta, y, { ev_opt });

    auto ev_release = oneapi::mkl::sparse::release_matrix_handle(main_queue, &handle, { ev_gemv });

    ev_release.wait_and_throw();

    //
    // Post Processing
    //

    fp *res = y;
    for (intType row = 0; row < nrows; row++) {
        fp tmp = set_fp_value(fp(0.0));
        for (intType i = ia[row]; i < ia[row + 1]; i++) {
            tmp += a[i] * x[ja[i]];
        }
        z[row] = alpha * tmp + beta * z[row];
    }

    fp diff = set_fp_value(fp(0.0));
    for (intType i = 0; i < nrows; i++) {
        diff += (z[i] - res[i]) * (z[i] - res[i]);
    }
    std::cout << "\n\t\t sparse::gemv residual:\n"
              << "\t\t\t" << diff << "\n\tFinished" << std::endl;

    free_vec(fp_ptr_vec, main_queue);
    free_vec(int_ptr_vec, main_queue);

    if (diff > 0)
        return 1;

    return 0;
}

//
// Description of example setup, apis used and supported floating point type
// precisions
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# Sparse Matrix-Vector Multiply Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# y = alpha * op(A) * x + beta * y" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A is a sparse matrix in CSR format, x and y are "
                 "dense vectors"
              << std::endl;
    std::cout << "# and alpha, beta are floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   sparse::gemv" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Device will be selected during runtime." << std::endl;
    std::cout << "# The environment variable SYCL_DEVICE_FILTER can be used to specify"
              << std::endl;
    std::cout << "# SYCL device" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example
//
int main(int /*argc*/, char ** /*argv*/) {
    print_example_banner();

    try {
        sycl::device dev = sycl::device();

        if (dev.is_gpu()) {
            std::cout << "Running Sparse BLAS GEMV USM example on GPU device." << std::endl;
            std::cout << "Device name is: " << dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running Sparse BLAS GEMV USM example on CPU device." << std::endl;
            std::cout << "Device name is: " << dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        std::cout << "Running with single precision real data type:" << std::endl;

        run_sparse_matrix_vector_multiply_example(dev);
        std::cout << "Sparse BLAS GEMV USM example ran OK." << std::endl;
    }
    catch (sycl::exception const &e) {
        std::cerr << "Caught synchronous SYCL exception during Sparse GEMV:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const &e) {
        std::cerr << "Caught std::exception during Sparse GEMV:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}
