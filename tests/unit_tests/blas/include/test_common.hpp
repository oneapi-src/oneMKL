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

#ifndef _TEST_COMMON_HPP__
#define _TEST_COMMON_HPP__

#include <complex>
#include <type_traits>

#include <CL/sycl.hpp>

namespace std {
static cl::sycl::half abs(cl::sycl::half v) {
    if (v < cl::sycl::half(0))
        return -v;
    else
        return v;
}
} // namespace std

// Complex helpers.
template <typename T>
struct complex_info {
    using real_type              = T;
    static const bool is_complex = false;
};

template <typename T>
struct complex_info<std::complex<T>> {
    using real_type              = T;
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

// Matrix helpers.
template <typename T>
constexpr T inner_dimension(onemkl::transpose trans, T m, T n) {
    return (trans == onemkl::transpose::nontrans) ? m : n;
}
template <typename T>
constexpr T outer_dimension(onemkl::transpose trans, T m, T n) {
    return (trans == onemkl::transpose::nontrans) ? n : m;
}
template <typename T>
constexpr T matrix_size(onemkl::transpose trans, T m, T n, T ldm) {
    return outer_dimension(trans, m, n) * ldm;
}

// SYCL buffer creation helper.
template <typename vec>
cl::sycl::buffer<typename vec::value_type, 1> make_buffer(const vec &v) {
    cl::sycl::buffer<typename vec::value_type, 1> buf(v.data(), cl::sycl::range<1>(v.size()));
    return buf;
}

// Reference helpers.
template <typename T>
struct ref_type_info {
    using type = T;
};
template <>
struct ref_type_info<std::complex<float>> {
    using type = std::complex<float>;
};
template <>
struct ref_type_info<std::complex<double>> {
    using type = std::complex<double>;
};
template <>
struct ref_type_info<int8_t> {
    using type = int8_t;
};
template <>
struct ref_type_info<uint8_t> {
    using type = uint8_t;
};
template <>
struct ref_type_info<int32_t> {
    using type = int32_t;
};

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
int8_t rand_scalar() {
    return std::rand() % 254 - 127;
}
template <>
int32_t rand_scalar() {
    return std::rand() % 256 - 128;
}
template <>
uint8_t rand_scalar() {
    return std::rand() % 128;
}

template <>
half rand_scalar() {
    return half(std::rand() % 32000) / half(32000) - half(0.5);
}

template <typename fp>
static fp rand_scalar(int mag) {
    fp tmp = fp(mag) + fp(std::rand()) / fp(RAND_MAX) - fp(0.5);
    if (std::rand() % 2)
        return tmp;
    else
        return -tmp;
}
template <typename fp>
static std::complex<fp> rand_complex_scalar(int mag) {
    return std::complex<fp>(rand_scalar<fp>(mag), rand_scalar<fp>(mag));
}
template <>
std::complex<float> rand_scalar(int mag) {
    return rand_complex_scalar<float>(mag);
}
template <>
std::complex<double> rand_scalar(int mag) {
    return rand_complex_scalar<double>(mag);
}

template <typename vec>
void rand_vector(vec &v, int n, int inc) {
    using fp    = typename vec::value_type;
    int abs_inc = std::abs(inc);

    v.resize(n * abs_inc);

    for (int i = 0; i < n; i++)
        v[i * abs_inc] = rand_scalar<fp>();
}

template <typename vec>
void print_matrix(vec &M, onemkl::transpose trans, int m, int n, int ld, char *name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (trans == onemkl::transpose::nontrans)
                std::cout << (double)M[i + j * ld] << " ";
            else
                std::cout << (double)M[j + i * ld] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename vec_src, typename vec_dest>
void copy_matrix(vec_src &src, onemkl::transpose trans, int m, int n, int ld, vec_dest &dest) {
    using T_data = typename vec_dest::value_type;
    dest.resize(matrix_size(trans, m, n, ld));
    if (trans == onemkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                dest[i + j * ld] = (T_data)src[i + j * ld];
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                dest[j + i * ld] = (T_data)src[j + i * ld];
    }
}

template <typename vec>
void rand_matrix(vec &M, onemkl::transpose trans, int m, int n, int ld) {
    using fp = typename vec::value_type;

    M.resize(matrix_size(trans, m, n, ld));

    if (trans == onemkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

template <typename fp>
void rand_matrix(fp *M, onemkl::transpose trans, int m, int n, int ld) {
    if (trans == onemkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

template <typename vec>
void rand_trsm_matrix(vec &M, onemkl::transpose trans, int m, int n, int ld) {
    using fp = typename vec::value_type;

    M.resize(matrix_size(trans, m, n, ld));

    if (trans == onemkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                if (i == j)
                    M[i + j * ld] = rand_scalar<fp>(10);
                else
                    M[i + j * ld] = rand_scalar<fp>();
            }
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                if (i == j)
                    M[j + i * ld] = rand_scalar<fp>(10);
                else
                    M[j + i * ld] = rand_scalar<fp>();
            }
    }
}

template <typename fp>
void rand_trsm_matrix(fp *M, onemkl::transpose trans, int m, int n, int ld) {
    if (trans == onemkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                if (i == j)
                    M[i + j * ld] = rand_scalar<fp>(10);
                else
                    M[i + j * ld] = rand_scalar<fp>();
            }
    }
    else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                if (i == j)
                    M[j + i * ld] = rand_scalar<fp>(10);
                else
                    M[j + i * ld] = rand_scalar<fp>();
            }
    }
}

// Correctness checking.
template <typename fp>
typename std::enable_if<!std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                              int error_mag) {
    using fp_real = typename complex_info<fp>::real_type;
    fp_real bound = (error_mag * num_components<fp>() * std::numeric_limits<fp_real>::epsilon());

    bool ok;

    fp_real aerr = std::abs(x - x_ref);
    fp_real rerr = aerr / std::abs(x_ref);
    ok           = (rerr <= bound) || (aerr <= bound);
    if (!ok)
        std::cout << "relative error = " << rerr << " absolute error = " << aerr
                  << " limit = " << bound << std::endl;
    return ok;
}

template <typename fp>
typename std::enable_if<std::is_integral<fp>::value, bool>::type check_equal(fp x, fp x_ref,
                                                                             int error_mag) {
    return (x == x_ref);
}

template <typename fp>
bool check_equal_trsm(fp x, fp x_ref, int error_mag) {
    using fp_real = typename complex_info<fp>::real_type;
    fp_real bound = std::max(fp_real(5e-5), (error_mag * num_components<fp>() *
                                             std::numeric_limits<fp_real>::epsilon()));
    bool ok;

    fp_real aerr = std::abs(x - x_ref);
    fp_real rerr = aerr / std::abs(x_ref);
    ok           = (rerr <= bound) || (aerr <= bound);
    if (!ok)
        std::cout << "relative error = " << rerr << " absolute error = " << aerr
                  << " limit = " << bound << std::endl;
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

template <typename vec1, typename vec2>
bool check_equal_vector(vec1 &v, vec2 &v_ref, int n, int inc, int error_mag, std::ostream &out) {
    int abs_inc = std::abs(inc);
    bool good   = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": DPC++ " << v[i * abs_inc]
                      << " vs. Reference " << v_ref[i * abs_inc] << std::endl;
            good = false;
        }
    }

    return good;
}

template <typename vec1, typename vec2>
bool check_equal_trsv_vector(vec1 &v, vec2 &v_ref, int n, int inc, int error_mag,
                             std::ostream &out) {
    int abs_inc = std::abs(inc);
    bool good   = true;

    for (int i = 0; i < n; i++) {
        if (!check_equal_trsm(v[i * abs_inc], v_ref[i * abs_inc], error_mag)) {
            int i_actual = (inc > 0) ? i : n - i;
            std::cout << "Difference in entry " << i_actual << ": DPC++ " << v[i * abs_inc]
                      << " vs. Reference " << v_ref[i * abs_inc] << std::endl;
            good = false;
        }
    }

    return good;
}

template <typename acc1, typename acc2>
bool check_equal_matrix(acc1 &M, acc2 &M_ref, int m, int n, int ld, int error_mag,
                        std::ostream &out) {
    bool good = true;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (!check_equal(M[i + j * ld], M_ref[i + j * ld], error_mag)) {
                out << "Difference in entry (" << i << ',' << j << "): DPC++ " << M[i + j * ld]
                    << " vs. Reference " << M_ref[i + j * ld] << std::endl;
                good = false;
            }
        }
    }

    return good;
}

template <typename fp>
bool check_equal_matrix(fp *M, fp *M_ref, int m, int n, int ld, int error_mag, std::ostream &out) {
    bool good = true;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (!check_equal(M[i + j * ld], M_ref[i + j * ld], error_mag)) {
                out << "Difference in entry (" << i << ',' << j << "): DPC++ " << M[i + j * ld]
                    << " vs. Reference " << M_ref[i + j * ld] << std::endl;
                good = false;
            }
        }
    }

    return good;
}

template <typename acc1, typename acc2>
bool check_equal_matrix(acc1 &M, acc2 &M_ref, onemkl::uplo upper_lower, int m, int n, int ld,
                        int error_mag, std::ostream &out) {
    bool good = true;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (((upper_lower == onemkl::uplo::upper) && (j >= i)) ||
                ((upper_lower == onemkl::uplo::lower) && (j <= i))) {
                if (!check_equal(M[i + j * ld], M_ref[i + j * ld], error_mag)) {
                    out << "Difference in entry (" << i << ',' << j << "): DPC++ " << M[i + j * ld]
                        << " vs. Reference " << M_ref[i + j * ld] << std::endl;
                    good = false;
                }
            }
        }
    }

    return good;
}

template <typename acc1, typename acc2>
bool check_equal_trsm_matrix(acc1 &M, acc2 &M_ref, int m, int n, int ld, int error_mag,
                             std::ostream &out) {
    bool good = true;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (!check_equal_trsm(M[i + j * ld], M_ref[i + j * ld], error_mag)) {
                out << "Difference in entry (" << i << ',' << j << "): DPC++ " << M[i + j * ld]
                    << " vs. Reference " << M_ref[i + j * ld] << std::endl;
                good = false;
            }
        }
    }

    return good;
}

#endif /* header guard */
