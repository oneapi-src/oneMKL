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
        pimpl_.reset(detail::create_commit(get_device_id(queue), queue, values));
    }

#ifdef ENABLE_MKLCPU_BACKEND
    void commit(backend_selector<backend::mklcpu> selector) {
        pimpl_.reset(mklcpu::create_commit(selector.get_queue(), values));
    }
#endif

#ifdef ENABLE_MKLGPU_BACKEND
    void commit(backend_selector<backend::mklgpu> selector) {
        // pimpl_.reset(mklgpu::create_commit<prec, dom>(selector.get_queue()));
    }
#endif

    sycl::queue& get_queue() {
        return queue_;
    }
private:
    sycl::queue queue_;
    std::unique_ptr<detail::commit_impl> pimpl_; // commit only

    std::int64_t rank_;
    std::vector<std::int64_t>  dimension_;

    // descriptor configuration values and structs
    void* handle_;
    oneapi::mkl::dft::dft_values values;
};

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::vector<std::int64_t> dimension) :
    dimension_(dimension),
    handle_(nullptr),
    rank_(dimension.size())
    {
        // TODO: initialize the device_handle, handle_buffer
        values.domain = dom;
        values.precision = prec;
        values.dimension = dimension_;
        values.rank = rank_;
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
                values.set_input_strides = true;
            case config_param::OUTPUT_STRIDES: {
                int64_t *strides = va_arg(vl, int64_t *);
                if (strides == nullptr) break;

                if (param == config_param::INPUT_STRIDES)
                    std::copy(strides, strides+rank_+1, std::back_inserter(values.input_strides));
                if (param == config_param::OUTPUT_STRIDES)
                    std::copy(strides, strides+rank_+1, std::back_inserter(values.output_strides));
                values.set_output_strides = true;
            } break;
            case config_param::FORWARD_SCALE:
                values.fwd_scale = va_arg(vl, double);
                values.set_fwd_scale = true;
                break;
            case config_param::BACKWARD_SCALE:
                values.bwd_scale = va_arg(vl, double);
                values.set_bwd_scale = true;
                break;
            case config_param::NUMBER_OF_TRANSFORMS:
                values.number_of_transforms = va_arg(vl, int64_t);
                values.set_number_of_transforms = true;
                break;
            case config_param::FWD_DISTANCE:
                values.fwd_dist = va_arg(vl, int64_t);
                values.set_fwd_dist = true;
                break;
            case config_param::BWD_DISTANCE:
                values.bwd_dist = va_arg(vl, int64_t);
                values.set_bwd_dist = true;
                break;
            case config_param::PLACEMENT:
                values.placement = va_arg(vl, config_value);
                values.set_placement = true;
                break;
            case config_param::COMPLEX_STORAGE:
                values.complex_storage = va_arg(vl, config_value);
                values.set_complex_storage = true;
                break;
            case config_param::CONJUGATE_EVEN_STORAGE:
                values.conj_even_storage = va_arg(vl, config_value);
                values.set_conj_even_storage = true;
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
