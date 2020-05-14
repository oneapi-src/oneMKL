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

#ifndef _FP16_HPP_
#define _FP16_HPP_

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace onemkl {
namespace mklcpu {

union float_raw {
    float f;
    uint32_t i;
};

static inline uint32_t float_to_raw(float f) {
    float_raw r;
    r.f = f;
    return r.i;
}

static inline float raw_to_float(uint32_t i) {
    float_raw r;
    r.i = i;
    return r.f;
}

namespace fp16_impl {

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

} // namespace fp16_impl

// fp16: seeeeemm'mmmmmmmm

struct fp16 {
    uint16_t raw;

    fp16(int raw_, bool) : raw(raw_) {}

    fp16() {}
    fp16(float f);
    fp16(double d) : fp16(float(d)) {}
    template <typename T>
    fp16(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr)
            : fp16(float(i)) {}

    inline operator float() const;

    fp16 operator+() const {
        return *this;
    }
    fp16 operator-() const {
        fp16 h = *this;
        h.raw ^= 0x8000;
        return h;
    }

    fp16 operator++() {
        return (*this = *this + 1);
    }
    fp16 operator++(int) {
        fp16 h = *this;
        ++*this;
        return h;
    }
    fp16 operator--() {
        return (*this = *this - 1);
    }
    fp16 operator--(int) {
        fp16 h = *this;
        --*this;
        return h;
    }

    friend float operator+(const fp16 &h1, const fp16 &h2) {
        return float(h1) + float(h2);
    }
    friend float operator-(const fp16 &h1, const fp16 &h2) {
        return float(h1) - float(h2);
    }
    friend float operator*(const fp16 &h1, const fp16 &h2) {
        return float(h1) * float(h2);
    }
    friend float operator/(const fp16 &h1, const fp16 &h2) {
        return float(h1) / float(h2);
    }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator+(const fp16 &h,
                                                                                      const T &o) {
        return float(h) + float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator-(const fp16 &h,
                                                                                      const T &o) {
        return float(h) - float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator*(const fp16 &h,
                                                                                      const T &o) {
        return float(h) * float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator/(const fp16 &h,
                                                                                      const T &o) {
        return float(h) / float(o);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator+(
        const T &o, const fp16 &h) {
        return float(o) + float(h);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator-(
        const T &o, const fp16 &h) {
        return float(o) - float(h);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator*(
        const T &o, const fp16 &h) {
        return float(o) * float(h);
    }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type operator/(
        const T &o, const fp16 &h) {
        return float(o) / float(h);
    }

    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator+(
        const fp16 &h, const T &o) {
        return float(h) + o;
    }
    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator-(
        const fp16 &h, const T &o) {
        return float(h) - o;
    }
    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator*(
        const fp16 &h, const T &o) {
        return float(h) * o;
    }
    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator/(
        const fp16 &h, const T &o) {
        return float(h) / o;
    }
    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator+(
        const T &o, const fp16 &h) {
        return o + float(h);
    }
    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator-(
        const T &o, const fp16 &h) {
        return o - float(h);
    }
    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator*(
        const T &o, const fp16 &h) {
        return o * float(h);
    }
    template <typename T>
    friend typename std::enable_if<fp16_impl::is_float_double<T>::value, T>::type operator/(
        const T &o, const fp16 &h) {
        return o / float(h);
    }
};

fp16::fp16(float f) {
    uint32_t i = float_to_raw(f);
    uint32_t s = i >> 31;
    uint32_t e = (i >> 23) & 0xFF;
    uint32_t m = i & 0x7FFFFF;

    uint32_t ss = s;
    uint32_t mm = m >> 13;
    uint32_t r  = m & 0x1FFF;
    uint32_t ee = 0;
    int32_t eee = (e - 127) + 15;

    if (e == 0) {
        // Denormal/zero floats all become zero.
        ee = 0;
        mm = 0;
    }
    else if (e == 0xFF) {
        // Preserve inf/nan.
        ee = 0x1F;
        if (m != 0 && mm == 0)
            mm = 1;
    }
    else if (eee > 0 && eee < 0x1F) {
        // Normal range. Perform round to even on mantissa.
        ee = eee;
        if (r > (0x1000 - (mm & 1))) {
            // Round up.
            mm++;
            if (mm == 0x400) {
                // Rounds up to next dyad (or inf).
                mm = 0;
                ee++;
            }
        }
    }
    else if (eee >= 0x1F) {
        // Overflow.
        ee = 0x1F;
        mm = 0;
    }
    else {
        // Underflow. Scale the input float, converting it
        //  into an equivalent denormal.
        float ff    = f * raw_to_float(0x01000000);
        uint32_t ii = float_to_raw(ff);
        ;
        ee = 0;
        mm = ii;
    }

    raw = (ss << 15) | (ee << 10) | mm;
}

inline fp16::operator float() const {
    uint32_t ss = raw >> 15;
    uint32_t ee = (raw >> 10) & 0x1F;
    uint32_t mm = raw & 0x3FF;

    uint32_t s   = ss;
    uint32_t eee = ee - 15 + 127;
    uint32_t m   = mm << 13;
    uint32_t e;

    if (ee == 0) {
        if (mm == 0)
            e = 0;
        else {
            // Half denormal -> float normal
            return (ss ? -1 : 1) * std::scalbn((float)mm, -24);
        }
    }
    else if (ee == 0x1F) {
        // inf/nan
        e = 0xFF;
    }
    else
        e = eee;

    uint32_t f = (s << 31) | (e << 23) | m;

    return raw_to_float(f);
}

} // namespace mklcpu
} // namespace onemkl

namespace std {

bool isfinite(onemkl::mklcpu::fp16 h) {
    return (~h.raw & 0x7C00);
}

onemkl::mklcpu::fp16 abs(onemkl::mklcpu::fp16 h) {
    onemkl::mklcpu::fp16 a = h;
    a.raw &= ~0x8000;
    return a;
}

onemkl::mklcpu::fp16 real(onemkl::mklcpu::fp16 h) {
    return h;
}

float imag(onemkl::mklcpu::fp16 h) {
    return 0.0f;
}

template <>
class numeric_limits<onemkl::mklcpu::fp16> {
public:
    static onemkl::mklcpu::fp16 min() {
        return onemkl::mklcpu::fp16(0x0100, false);
    }
    static onemkl::mklcpu::fp16 lowest() {
        return onemkl::mklcpu::fp16(0xFBFF, false);
    }
    static onemkl::mklcpu::fp16 max() {
        return onemkl::mklcpu::fp16(0x7BFF, false);
    }
    static onemkl::mklcpu::fp16 epsilon() {
        return onemkl::mklcpu::fp16(0x1400, false);
    }
    static onemkl::mklcpu::fp16 round_error() {
        return onemkl::mklcpu::fp16(0x3800, false);
    } // 0.5ulp
    static onemkl::mklcpu::fp16 infinity() {
        return onemkl::mklcpu::fp16(0x7C00, false);
    }
    static onemkl::mklcpu::fp16 quiet_NaN() {
        return onemkl::mklcpu::fp16(0x7D00, false);
    }
    static onemkl::mklcpu::fp16 signaling_NaN() {
        return onemkl::mklcpu::fp16(0x7E00, false);
    }
    static onemkl::mklcpu::fp16 denorm_min() {
        return onemkl::mklcpu::fp16(0x0001, false);
    }
};

} // namespace std

#endif //_FP16_HPP_
