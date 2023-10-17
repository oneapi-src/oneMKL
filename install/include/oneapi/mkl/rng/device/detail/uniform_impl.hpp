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

#ifndef _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
#define _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_

namespace oneapi::mkl::rng::device::detail {

template <typename Type, typename Method>
class distribution_base<oneapi::mkl::rng::device::uniform<Type, Method>> {
public:
    struct param_type {
        param_type(Type a, Type b) : a_(a), b_(b) {}
        Type a_;
        Type b_;
    };

    distribution_base(Type a, Type b) : a_(a), b_(b) {
#ifndef __SYCL_DEVICE_ONLY__
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

    param_type param() const {
        return param_type(a_, b_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.a_ >= pt.b_) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
        a_ = pt.a_;
        b_ = pt.b_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, Type,
                                  sycl::vec<Type, EngineType::vec_size>>::type {
        using OutType = typename std::conditional<EngineType::vec_size == 1, Type,
                                                  sycl::vec<Type, EngineType::vec_size>>::type;
        using FpType =
            typename std::conditional<std::is_same<Method, uniform_method::accurate>::value, double,
                                      float>::type;
        OutType res;
        if constexpr (std::is_integral<Type>::value) {
            if constexpr (EngineType::vec_size == 1) {
                FpType res_fp = engine.generate(static_cast<FpType>(a_), static_cast<FpType>(b_));
                res_fp = sycl::floor(res_fp);
                res = static_cast<Type>(res_fp);
                return res;
            }
            else {
                sycl::vec<FpType, EngineType::vec_size> res_fp;
                res_fp = engine.generate(static_cast<FpType>(a_), static_cast<FpType>(b_));
                res_fp = sycl::floor(res_fp);
                res = res_fp.template convert<Type>();
                return res;
            }
        }
        else {
            res = engine.generate(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
                res = sycl::fmax(res, a_);
                res = sycl::fmin(res, b_);
            }
        }

        return res;
    }

    template <typename EngineType>
    Type generate_single(EngineType& engine) {
        using FpType =
            typename std::conditional<std::is_same<Method, uniform_method::accurate>::value, double,
                                      float>::type;
        Type res;
        if constexpr (std::is_integral<Type>::value) {
            FpType res_fp =
                engine.generate_single(static_cast<FpType>(a_), static_cast<FpType>(b_));
            res_fp = sycl::floor(res_fp);
            res = static_cast<Type>(res_fp);
            return res;
        }
        else {
            res = engine.generate_single(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
                res = sycl::fmax(res, a_);
                res = sycl::fmin(res, b_);
            }
        }

        return res;
    }

    Type a_;
    Type b_;
};

} // namespace oneapi::mkl::rng::device::detail

#endif // _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
