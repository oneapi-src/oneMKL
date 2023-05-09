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
#include <stdexcept>
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

inline std::size_t cast_unsigned(std::int64_t i) {
    if (i < 0) {
        throw std::runtime_error("Unexpected negative value");
    }
    return static_cast<std::size_t>(i);
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
    const auto abs_bound = static_cast<fp_real>(abs_error_mag) * epsilon;
    const auto rel_bound = static_cast<fp_real>(rel_error_mag) * epsilon;

    const auto aerr = std::abs(x - x_ref);
    const auto rerr = aerr / std::abs(x_ref);
    const bool ok = (rerr <= rel_bound) || (aerr <= abs_bound);
    if (!ok) {
        out << "Mismatching results: actual = " << x << " vs. reference = " << x_ref << "\n";
        out << " relative error = " << rerr << " absolute error = " << aerr
            << " relative bound = " << rel_bound << " absolute bound = " << abs_bound << "\n";
    }
    return ok;
}

template <typename vec1, typename vec2>
bool check_equal_vector(vec1 &&v, vec2 &&v_ref, std::size_t n, double abs_error_mag,
                        double rel_error_mag, std::ostream &out) {
    constexpr int max_print = 20;
    int count = 0;
    bool good = true;

    for (std::size_t i = 0; i < n; ++i) {
        // Allow to convert the unsigned index `i` to a signed one to keep this function generic and allow for `v` and `v_ref` to be a vector, a pointer or a random access iterator.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
        auto res = v[i];
        auto ref = v_ref[i];
#pragma clang diagnostic pop
        if (!check_equal(res, ref, abs_error_mag, rel_error_mag, out)) {
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
void rand_vector(vec &v, std::size_t n) {
    using fp = typename vec::value_type;
    v.resize(n);
    for (std::size_t i = 0; i < n; i++) {
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

// is it assumed that the unused elements of the array are ignored
inline std::array<std::int64_t, 4> get_conjugate_even_complex_strides(
    const std::vector<std::int64_t> &sizes) {
    switch (sizes.size()) {
        case 1: return { 0, 1 };
        case 2: return { 0, sizes[1] / 2 + 1, 1 };
        case 3: return { 0, sizes[1] * (sizes[2] / 2 + 1), (sizes[2] / 2 + 1), 1 };
        default:
            throw oneapi::mkl::unimplemented(
                "dft/test_common", __FUNCTION__,
                "not implemented for " + std::to_string(sizes.size()) + " dimensions");
            return {};
    }
}

// is it assumed that the unused elements of the array are ignored
inline std::array<std::int64_t, 4> get_default_strides(const std::vector<std::int64_t> &sizes) {
    if (sizes.size() > 3) {
        throw oneapi::mkl::unimplemented(
            "dft/test_common", __FUNCTION__,
            "not implemented for " + std::to_string(sizes.size()) + " dimensions");
    }

    switch (sizes.size()) {
        case 1: return { 0, 1 };
        case 2: return { 0, sizes[1], 1 };
        case 3: return { 0, sizes[1] * sizes[2], sizes[2], 1 };
        default:
            throw oneapi::mkl::unimplemented(
                "dft/test_common", __FUNCTION__,
                "not implemented for " + std::to_string(sizes.size()) + " dimensions");
            return {};
    }
}

struct DFTParams {
    std::vector<std::int64_t> sizes;
    std::int64_t batches;
};

class DFTParamsPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<sycl::device *, DFTParams>> dev) const {
        auto [device, params] = dev.param;
        auto [sizes, batches] = params;
        std::string info_name;

        assert(sizes.size() > 0);
        info_name.append("sizes_");

        // intersperse dimensions with "x"
        std::for_each(sizes.begin(), sizes.end() - 1,
                      [&info_name](auto s) { info_name.append(std::to_string(s)).append("x"); });
        info_name.append(std::to_string(sizes.back()));

        info_name.append("_batches_").append(std::to_string(batches));

        std::string dev_name = device->get_info<sycl::info::device::name>();
        std::for_each(dev_name.begin(), dev_name.end(), [](auto &c) {
            if (!isalnum(c))
                c = '_';
        });

        info_name.append("_").append(dev_name);

        return info_name;
    }
};

#endif //ONEMKL_TEST_COMMON_HPP
