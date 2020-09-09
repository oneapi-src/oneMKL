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

#ifndef _ONEMKL_RNG_DISTRIBUTIONS_HPP_
#define _ONEMKL_RNG_DISTRIBUTIONS_HPP_

#include <cstdint>
#include <limits>
#include <CL/sycl.hpp>

#include "oneapi/mkl/exceptions.hpp"

namespace oneapi {
namespace mkl {
namespace rng {

// Class template oneapi::mkl::rng::uniform
//
// Represents continuous and discrete uniform random number distribution
//
// Supported types:
//      float
//      double
//      std::int32_t
//
// Supported methods:
//      oneapi::mkl::rng::uniform_method::standard
//      oneapi::mkl::rng::uniform_method::accurate - for float and double types only
//
// Input arguments:
//      a - left bound. 0.0 by default
//      b - right bound. 1.0 by default (std::numeric_limits<std::int32_t>::max() for std::int32_t)

namespace uniform_method {
struct standard {};
struct accurate {};
using by_default = standard;
} // namespace uniform_method

template <typename Type = float, typename Method = uniform_method::by_default>
class uniform {
public:
    static_assert(std::is_same<Method, uniform_method::standard>::value ||
                      (std::is_same<Method, uniform_method::accurate>::value &&
                       !std::is_same<Type, std::int32_t>::value),
                  "rng uniform distribution method is incorrect");

    static_assert(std::is_same<Type, float>::value || std::is_same<Type, double>::value,
                  "rng uniform distribution type is not supported");

    using method_type = Method;
    using result_type = Type;

    uniform() : uniform(static_cast<Type>(0.0f), static_cast<Type>(1.0f)) {}

    explicit uniform(Type a, Type b) : a_(a), b_(b) {
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform distribution",
                                                "parameters are incorrect, a >= b");
        }
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

private:
    Type a_;
    Type b_;
};

template <typename Method>
class uniform<std::int32_t, Method> {
public:
    using method_type = Method;
    using result_type = std::int32_t;

    uniform() : uniform(0, std::numeric_limits<std::int32_t>::max()) {}

    explicit uniform(std::int32_t a, std::int32_t b) : a_(a), b_(b) {
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform distribution",
                                                "parameters are incorrect, a >= b");
        }
    }

    std::int32_t a() const {
        return a_;
    }

    std::int32_t b() const {
        return b_;
    }

private:
    std::int32_t a_;
    std::int32_t b_;
};

template <typename UIntType = std::uint32_t>
class bits {
public:
    static_assert(std::is_same<UIntType, std::uint32_t>::value,
                  "rng uniform distribution type is not supported");
    using result_type = UIntType;
};

namespace gaussian_method {
struct icdf {};
struct box_muller2 {};
using by_default = box_muller2;
} // namespace gaussian_method

template <typename RealType = float, typename Method = gaussian_method::by_default>
class gaussian {
public:
    static_assert(std::is_same<Method, gaussian_method::icdf>::value ||
                      std::is_same<Method, gaussian_method::box_muller2>::value,
                  "rng gaussian distribution method is incorrect");

    static_assert(std::is_same<RealType, float>::value || std::is_same<RealType, double>::value,
                  "rng gaussian distribution type is not supported");

    using method_type = Method;
    using result_type = RealType;

    gaussian() : gaussian(static_cast<RealType>(0.0), static_cast<RealType>(1.0)) {}

    explicit gaussian(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {
        if (stddev <= static_cast<RealType>(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "gaussian distribution",
                                                "stddev parameter is incorrect, stddev <= 0.0");
        }
    }

    RealType mean() const {
        return mean_;
    }

    RealType stddev() const {
        return stddev_;
    }

private:
    RealType mean_;
    RealType stddev_;
};

} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_RNG_DISTRIBUTIONS_HPP_
