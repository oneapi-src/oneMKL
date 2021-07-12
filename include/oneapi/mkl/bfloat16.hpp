/*******************************************************************************
* Copyright 2018-2021 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#ifndef _BFLOAT16_HPP__
#define _BFLOAT16_HPP__

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace oneapi {
namespace mkl {

namespace bfloat16_impl {

template <typename T>
struct is_float_double {
    static constexpr bool value = false;
};
template <>
struct is_float_double<float> {
    static constexpr bool value = true;
};
template <>
struct is_float_double<double> {
    static constexpr bool value = true;
};

union float_raw {
    float f;
    std::uint32_t i;
};

static inline std::uint32_t float_to_raw(float f) {
    float_raw r;
    r.f = f;
    return r.i;
}

static inline float raw_to_float(std::uint32_t i) {
    float_raw r;
    r.i = i;
    return r.f;
}

} /* namespace bfloat16_impl */

struct bfloat16 {
    std::uint16_t raw;

    bfloat16(int raw_, bool) : raw(raw_) {}

    bfloat16() = default;
    inline bfloat16(float f);
    bfloat16(double d) : bfloat16(float(d)) {}
    template <typename T>
    bfloat16(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr)
            : bfloat16(float(i)) {}

    inline operator float() const;

    bfloat16 operator+() const {
        return *this;
    }
    bfloat16 operator-() const {
        bfloat16 h = *this;
        h.raw ^= 0x8000;
        return h;
    }

    bfloat16 operator++() {
        return (*this = *this + 1);
    }
    bfloat16 operator++(int) {
        bfloat16 h = *this;
        ++*this;
        return h;
    }
    bfloat16 operator--() {
        return (*this = *this - 1);
    }
    bfloat16 operator--(int) {
        bfloat16 h = *this;
        --*this;
        return h;
    }

    friend float operator+(const bfloat16 &h1, const bfloat16 &h2) {
        return float(h1) + float(h2);
    }
    friend float operator-(const bfloat16 &h1, const bfloat16 &h2) {
        return float(h1) - float(h2);
    }
    friend float operator*(const bfloat16 &h1, const bfloat16 &h2) {
        return float(h1) * float(h2);
    }
    friend float operator/(const bfloat16 &h1, const bfloat16 &h2) {
        return float(h1) / float(h2);
    }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator+(
        const bfloat16 &h, const T &o) {
        return float(h) + float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator-(
        const bfloat16 &h, const T &o) {
        return float(h) - float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator*(
        const bfloat16 &h, const T &o) {
        return float(h) * float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator/(
        const bfloat16 &h, const T &o) {
        return float(h) / float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator+(
        const T &o, const bfloat16 &h) {
        return float(o) + float(h);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator-(
        const T &o, const bfloat16 &h) {
        return float(o) - float(h);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator*(
        const T &o, const bfloat16 &h) {
        return float(o) * float(h);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator/(
        const T &o, const bfloat16 &h) {
        return float(o) / float(h);
    }

    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator+(
        const bfloat16 &h, const T &o) {
        return float(h) + o;
    }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator-(
        const bfloat16 &h, const T &o) {
        return float(h) - o;
    }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator*(
        const bfloat16 &h, const T &o) {
        return float(h) * o;
    }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator/(
        const bfloat16 &h, const T &o) {
        return float(h) / o;
    }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator+(
        const T &o, const bfloat16 &h) {
        return o + float(h);
    }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator-(
        const T &o, const bfloat16 &h) {
        return o - float(h);
    }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator*(
        const T &o, const bfloat16 &h) {
        return o * float(h);
    }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type operator/(
        const T &o, const bfloat16 &h) {
        return o / float(h);
    }

    template <typename T>
    bfloat16 operator+=(const T &o) {
        return *this = bfloat16(*this + o);
    }
    template <typename T>
    bfloat16 operator-=(const T &o) {
        return *this = bfloat16(*this - o);
    }
    template <typename T>
    bfloat16 operator*=(const T &o) {
        return *this = bfloat16(*this * o);
    }
    template <typename T>
    bfloat16 operator/=(const T &o) {
        return *this = bfloat16(*this / o);
    }
};

bfloat16::bfloat16(float f) {
    raw = bfloat16_impl::float_to_raw(f) >> 16; // RTZ
}

inline bfloat16::operator float() const {
    return bfloat16_impl::raw_to_float(raw << 16);
}

} /* namespace mkl */
} // namespace oneapi

#endif /* _BFLOAT16_HPP__ */
