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

#ifndef ONEMKL_TEST_COMMON_HPP
#define ONEMKL_TEST_COMMON_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

template <typename T>
struct complex_info {
    using real_type = T;
    static const bool is_complex = false;
};

template <typename T>
struct complex_info<std::complex<T>> {
    using real_type = T;
    static const bool is_complex = true;
};

template <typename T>
constexpr bool is_complex() {
    return complex_info<T>::is_complex;
}
template <typename T>
constexpr int num_components() {
    return is_complex<T>() ? 2 : 1;
}

template <typename fp>
typename std::enable_if<!std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                              int error_mag) {
    using fp_real = typename complex_info<fp>::real_type;
    fp_real bound = (error_mag * num_components<fp>() * std::numeric_limits<fp_real>::epsilon());

    bool ok;

    fp_real aerr = std::abs(x - x_ref);
    fp_real rerr = aerr / std::abs(x_ref);
    ok = (rerr <= bound) || (aerr <= bound);
    if (!ok) {
        std::cout << "relative error = " << rerr << " absolute error = " << aerr
                  << " limit = " << bound << std::endl;
    }
    return ok;
}

template <typename fp>
bool check_equal(fp x, fp x_ref, int error_mag, std::ostream &out) {
    bool good = check_equal(x, x_ref, error_mag);

    if (!good) {
        out << "Difference in result: DPC++ " << x << " vs. Reference " << x_ref << std::endl;
    }
    return good;
}

template <typename fp>
bool check_equal_vector(fp *v, fp *v_ref, int n, int inc, int error_mag, std::ostream &out) {
    constexpr size_t max_print = 20;
    int abs_inc = std::abs(inc), count = 0;
    bool good = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": DPC++ " << v[i * abs_inc]
                      << " vs. Reference " << v_ref[i * abs_inc] << std::endl;
            good = false;
            count++;
            if (count > max_print) {
                return good;
            }
        }
    }

    return good;
}

template <typename vec1, typename vec2>
bool check_equal_vector(vec1 &v, vec2 &v_ref, int n, int inc, int error_mag, std::ostream &out) {
    constexpr size_t max_print = 20;
    int abs_inc = std::abs(inc), count = 0;
    bool good = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": DPC++ " << v[i * abs_inc]
                      << " vs. Reference " << v_ref[i * abs_inc] << std::endl;
            good = false;
            count++;
            if (count > max_print) {
                return good;
            }
        }
    }

    return good;
}

// Random initialization.
template <typename fp>
static fp rand_scalar() {
    return fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
}
template <typename fp>
static std::complex<fp> rand_complex_scalar() {
    return std::complex<fp>(rand_scalar<fp>(), rand_scalar<fp>());
}
template <>
std::complex<float> rand_scalar() {
    return rand_complex_scalar<float>();
}
template <>
std::complex<double> rand_scalar() {
    return rand_complex_scalar<double>();
}

template <>
int32_t rand_scalar() {
    return std::rand() % 256 - 128;
}

template <typename fp>
void rand_vector(fp *v, int n, int inc) {
    int abs_inc = std::abs(inc);
    for (int i = 0; i < n; i++) {
        v[i * abs_inc] = rand_scalar<fp>();
    }
}

template <typename vec>
void rand_vector(vec &v, int n, int inc) {
    using fp = typename vec::value_type;
    int abs_inc = std::abs(inc);

    v.resize(n * abs_inc);

    for (int i = 0; i < n; i++) {
        v[i * abs_inc] = rand_scalar<fp>();
    }
}

// Catch asynchronous exceptions.
auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
            print_error_code(e);
        }
    }
};

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
void commit_descriptor(oneapi::mkl::dft::descriptor<precision, domain> &descriptor,
                       sycl::queue queue) {
#ifdef CALL_RT_API
    descriptor.commit(queue);
#else
    TEST_RUN_CT_SELECT_NO_ARGS(queue, descriptor.commit);
#endif
}

template <typename T, int D>
void copy_to_host(sycl::queue sycl_queue, sycl::buffer<T, D> &buffer_in, std::vector<T> &host_out) {
    sycl_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor accessor = buffer_in.get_access(cgh);
        cgh.copy(accessor, host_out.data());
    });
    sycl_queue.wait();
}

template <typename T, int D>
void copy_to_device(sycl::queue sycl_queue, std::vector<T> &host_in,
                    sycl::buffer<T, D> &buffer_out) {
    sycl_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor accessor = buffer_out.get_access(cgh);
        cgh.copy(host_in.data(), accessor);
    });
    sycl_queue.wait();
}

class DimensionsDeviceNamePrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<sycl::device *, std::int64_t>> dev) const {
        std::string size = "size_" + std::to_string(std::get<1>(dev.param));
        std::string dev_name = std::get<0>(dev.param)->get_info<sycl::info::device::name>();
        for (std::string::size_type i = 0; i < dev_name.size(); ++i) {
            if (!isalnum(dev_name[i]))
                dev_name[i] = '_';
        }
        std::string info_name = (size.append("_")).append(dev_name);
        return info_name;
    }
};

#endif //ONEMKL_TEST_COMMON_HPP
