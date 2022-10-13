#include <iostream>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/types.hpp"

#include "oneapi/mkl/dft/descriptor.hpp"
#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::vector<std::int64_t> dimension) :
    dimension_(dimension),
    handle_(nullptr),
    rank_(dimension.size())
    {
        // TODO: initialize the device_handle, handle_buffer
        auto handle = reinterpret_cast<DFTI_DESCRIPTOR_HANDLE>(handle_);
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
        switch (param) {
            case config_param::INPUT_STRIDES:
            case config_param::OUTPUT_STRIDES: {
                int64_t *strides = va_arg(vl, int64_t *);
                if (strides == nullptr) break;

                if (param == config_param::INPUT_STRIDES)
                    std::copy(strides, strides+rank_+1, std::back_inserter(values.input_strides));
                if (param == config_param::OUTPUT_STRIDES)
                    std::copy(strides, strides+rank_+1, std::back_inserter(values.output_strides));
            } break;
            case config_param::FORWARD_SCALE:
                values.fwd_scale = va_arg(vl, double);
                break;
            case config_param::BACKWARD_SCALE:
                values.bwd_scale = va_arg(vl, double);
                break;
            case config_param::NUMBER_OF_TRANSFORMS:
                values.number_of_transform = va_arg(vl, int64_t);
                break;
            case config_param::FWD_DISTANCE:
                values.fwd_dist = va_arg(vl, int64_t);
                break;
            case config_param::BWD_DISTANCE:
                values.bwd_dist = va_arg(vl, int64_t);
                break;
            case config_param::PLACEMENT:
                values.placement = va_arg(vl, config_value);
                break;
            case config_param::COMPLEX_STORAGE:
                values.complex_storage = va_arg(vl, config_value);
                break;
            case config_param::CONJUGATE_EVEN_STORAGE:
                values.conj_even_storage = va_arg(vl, config_value);
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

template class descriptor<precision::SINGLE, domain::COMPLEX>;
template class descriptor<precision::SINGLE, domain::REAL>;
template class descriptor<precision::DOUBLE, domain::COMPLEX>;
template class descriptor<precision::DOUBLE, domain::REAL>;

} // namespace dft
} // namespace mkl
} // namespace oneapi
