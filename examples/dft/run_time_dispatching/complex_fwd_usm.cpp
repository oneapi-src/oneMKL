
// stl includes
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// oneMKL/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"

// local includes
#include "example_helper.hpp"

constexpr int SUCCESS = 0;
constexpr int FAILURE = 1;
constexpr double TWOPI = 6.2831853071795864769;

void run_uniform_example(const sycl::device& dev) {

    int N = 16;
    int harmonic = 5;
    int buffer_result = FAILURE;
    int usm_result = FAILURE;
    int result = FAILURE;

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception during generation:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    sycl::queue queue(dev, exception_handler);

    double *x_usm = (double*) malloc_shared(N*2*sizeof(double), queue.get_device(), queue.get_context());

    oneapi::mkl::dft::descriptor<
        oneapi::mkl::dft::precision::DOUBLE,
        oneapi::mkl::dft::domain::COMPLEX
    > desc(N);
}

//
// Description of example setup, APIs used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout
        << "# DFTI complex in-place forward transform for USM/Buffer API's example: "
        << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using APIs:" << std::endl;
    std::cout << "#   USM/BUffer forward complex in-place" << std::endl;
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
// Main entry point for example.
//

int main(int argc, char** argv) {
    print_example_banner();

    try {
        sycl::device my_dev((sycl::default_selector()));

        if (my_dev.is_gpu()) {
            std::cout << "Running DFT complex forward example on GPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running DFT complex forward example on CPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        std::cout << "Running with single precision real data type:" << std::endl;

        run_uniform_example(my_dev);
        std::cout << "DFIT example ran OK" << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        std::cerr << "Caught std::exception during generation:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }
    return 0;
}
