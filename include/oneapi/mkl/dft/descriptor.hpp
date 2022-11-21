/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef _ONEMKL_DFT_DESCRIPTOR_HPP_
#define _ONEMKL_DFT_DESCRIPTOR_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/dft_loader.hpp"

#ifdef ENABLE_MKLCPU_BACKEND
#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"
#endif
#ifdef ENABLE_MKLGPU_BACKEND
#include "oneapi/mkl/dft/detail/mklgpu/onemkl_dft_mklgpu.hpp"
#endif
namespace oneapi {
namespace mkl {
namespace dft {

template <precision prec, domain dom>
class descriptor {
public:
    // Syntax for 1-dimensional DFT
    descriptor(std::int64_t length);

    // Syntax for d-dimensional DFT
    descriptor(std::vector<std::int64_t> dimensions);

    ~descriptor();

    void set_value(config_param param, ...);

    void get_value(config_param param, ...);

    void commit(sycl::queue& queue) {
        pimpl_.reset(detail::create_commit(get_device_id(queue), queue, values_));
    }

#ifdef ENABLE_MKLCPU_BACKEND
    void commit(backend_selector<backend::mklcpu> selector) {
        pimpl_.reset(mklcpu::create_commit(selector.get_queue(), values_));
    }
#endif

#ifdef ENABLE_MKLGPU_BACKEND
    void commit(backend_selector<backend::mklgpu> selector) {
        // pimpl_.reset(mklgpu::create_commit<prec, dom>(selector.get_queue()));
    }
#endif
private:
    std::unique_ptr<detail::commit_impl> pimpl_; // commit only

    std::int64_t rank_;
    std::vector<std::int64_t>  dimensions_;

    // descriptor configuration values_ and structs
    void* handle_;
    oneapi::mkl::dft::dft_values values_;
};

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::vector<std::int64_t> dimensions) :
    dimensions_(dimensions),
    handle_(nullptr),
    rank_(dimensions.size())
    {
        // Compute default strides.
        std::vector<std::int64_t> defaultStrides(rank_, 1);
        for(int i = rank_ - 1; i < 0; --i){
            defaultStrides[i] = defaultStrides[i - 1] * dimensions_[i];
        }
        defaultStrides[0] = 0;
        values_.input_strides = defaultStrides;
        values_.output_strides = defaultStrides;
        values_.bwd_scale = 1.0;
        values_.fwd_scale = 1.0;
        values_.number_of_transforms = 1;
        values_.fwd_dist = 1;
        values_.bwd_dist = 1;
        values_.placement = config_value::INPLACE;
        values_.complex_storage = config_value::COMPLEX_COMPLEX;
        values_.conj_even_storage = config_value::COMPLEX_COMPLEX; 
        values_.dimensions = dimensions_;
        values_.rank = rank_;
        values_.domain = dom;
        values_.precision = prec;
    }

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::int64_t length) :
    descriptor<prec, dom>(std::vector<std::int64_t>{length}) {}

template <precision prec, domain dom>
descriptor<prec, dom>::~descriptor() {
    // call DftiFreeDescriptor
}

// impliment error class
template <precision prec, domain dom>
void descriptor<prec, dom>::set_value(config_param param, ...) {
        int err = 0;
        va_list vl;
        va_start(vl, param);
        printf("oneapi interface set_value\n");
        switch (param) {
            case config_param::INPUT_STRIDES:
                [[fallthrough]];
            case config_param::OUTPUT_STRIDES: {
                int64_t *strides = va_arg(vl, int64_t *);
                if (strides == nullptr) break;

                if (param == config_param::INPUT_STRIDES)
                    std::copy(strides, strides+rank_+1, std::back_inserter(values_.input_strides));
                if (param == config_param::OUTPUT_STRIDES)
                    std::copy(strides, strides+rank_+1, std::back_inserter(values_.output_strides));
            } break;
            case config_param::FORWARD_SCALE:
                values_.fwd_scale = va_arg(vl, double);
                break;
            case config_param::BACKWARD_SCALE:
                values_.bwd_scale = va_arg(vl, double);
                break;
            case config_param::NUMBER_OF_TRANSFORMS:
                values_.number_of_transforms = va_arg(vl, int64_t);
                break;
            case config_param::FWD_DISTANCE:
                values_.fwd_dist = va_arg(vl, int64_t);
                break;
            case config_param::BWD_DISTANCE:
                values_.bwd_dist = va_arg(vl, int64_t);
                break;
            case config_param::PLACEMENT:
                values_.placement = va_arg(vl, config_value);
                break;
            case config_param::COMPLEX_STORAGE:
                values_.complex_storage = va_arg(vl, config_value);
            case config_param::CONJUGATE_EVEN_STORAGE:
                values_.conj_even_storage = va_arg(vl, config_value);
                break;

            default: err = 1;
        }
        va_end(vl);
}

template <precision prec, domain dom>
void descriptor<prec, dom>::get_value(config_param param, ...) {
    int err = 0;
    va_list vl;
    va_start(vl, param);
    switch (param)
    {
    default: break;
    }
    va_end(vl);
}

} //namespace dft
} //namespace mkl
} //namespace oneapi


#endif // _ONEMKL_DFT_DESCRIPTOR_HPP_
