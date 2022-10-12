#ifndef _ONEMKL_DFT_DESCRIPTOR_IMPL_HPP_
#define _ONEMKL_DFT_DESCRIPTOR_IMPL_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"
#include "oneapi/mkl/dft/types.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

class descriptor_impl {
public:
    descriptor_impl(std::size_t length) : length_(length) {}

    descriptor_impl(std::vector<std::int64_t> dimension) : dimension_(dimension) {}

    descriptor_impl(const descriptor_impl& other) : length_(other.length_) {}

    void set_value(config_param param, ...) {
        int err = 0;
        va_list vl;
        va_start(vl, param);
        switch (param)
        {
        case config_param::INPUT_STRIDES:
            // values.input_strides = va_arg(vl, std::vector<int64_t>);
            break;
        case config_param::OUTPUT_STRIDES:
            // values.output_strides = va_arg(vl, std::vector<int64_t>);
            break;
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

    virtual descriptor_impl* copy_state() = 0;

    virtual ~descriptor_impl() {}

    sycl::queue& get_queue() {
        return queue_;
    }

protected:
    sycl::queue queue_;
    std::size_t length_;
    std::vector<std::int64_t>  dimension_;

    // descriptor configuration values and structs
    oneapi::mkl::dft::precision prec_;
    oneapi::mkl::dft::domain dom_;
    oneapi::mkl::dft::dft_values values;
};

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DFT_DESCRIPTOR_IMPL_HPP_

