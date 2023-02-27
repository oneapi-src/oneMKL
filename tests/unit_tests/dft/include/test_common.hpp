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

template <typename fp>
bool check_equal(fp x, fp x_ref, double abs_error_mag, double rel_error_mag, std::ostream &out) {
    using fp_real = typename complex_info<fp>::real_type;
    static_assert(std::is_floating_point_v<fp_real>,
                  "Expected floating-point real or complex type.");

    const fp_real epsilon = []() {
        if constexpr (sizeof(double) == sizeof(long double) && std::is_same_v<fp_real, double>) {
            // The reference DFT uses long double to maintain accuracy
            // when this isn't possible, lower the accuracy requirements
            return 1e-12;
        }
        else {
            return std::numeric_limits<fp_real>::epsilon();
        }
    }();
    const fp_real abs_bound = abs_error_mag * epsilon;
    const fp_real rel_bound = rel_error_mag * epsilon;

    const auto aerr = std::abs(x - x_ref);
    const auto rerr = aerr / std::abs(x_ref);
    const bool ok = (rerr <= rel_bound) || (aerr <= abs_bound);
    if (!ok) {
        out << "Mismatching results: actual = " << x << " vs. reference = " << x_ref << "\n";
        out << " relative error = " << rerr
            << " absolute error = " << aerr
            << " relative bound = " << rel_bound
            << " absolute bound = " << abs_bound
            << "\n";
    }
    return ok;
}

template <typename vec1, typename vec2>
bool check_equal_vector(vec1 &&v, vec2 &&v_ref, int n, double abs_error_mag, double rel_error_mag, std::ostream &out) {
    constexpr int max_print = 20;
    int count = 0;
    bool good = true;

    for (std::size_t i = 0; i < n; ++i) {
        if (!check_equal(v[i], v_ref[i], abs_error_mag, rel_error_mag, out)) {
            out << " at index i =" << i << "\n";
            good = false;
            ++count;
            if (count > max_print) {
                return good;
            }
        }
    }

    return good;
}

// Random initialization.
template <typename t>
inline t rand_scalar() {
    if constexpr (std::is_same_v<t, int32_t>) {
        return std::rand() % 256 - 128;
    }
    else if constexpr (std::is_floating_point_v<t>) {
        return t(std::rand()) / t(RAND_MAX) - t(0.5);
    }
    else {
        static_assert(complex_info<t>::is_complex, "unexpect type in rand_scalar");
        using fp = typename complex_info<t>::real_type;
        return t(rand_scalar<fp>(), rand_scalar<fp>());
    }
}

template <typename vec>
void rand_vector(vec &v, int n) {
    using fp = typename vec::value_type;
    v.resize(n);
    for (int i = 0; i < n; i++) {
        v[i] = rand_scalar<fp>();
    }
}

// Catch asynchronous exceptions.
auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
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
        return size.append("_").append(dev_name);
    }
};

#endif //ONEMKL_TEST_COMMON_HPP
