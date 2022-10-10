#ifndef _ONEMKL_DFT_DESCRIPTOR_IMPL_HPP_
#define _ONEMKL_DFT_DESCRIPTOR_IMPL_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

class descriptor_impl {
public:
    descriptor_impl(std::size_t length) : length_(length) {}

    descriptor_impl(const descriptor_impl& other) : length_(other.length_) {}

    virtual descriptor_impl* copy_state() = 0;

    virtual ~descriptor_impl() {}

    sycl::queue& get_queue() {
        return queue_;
    }

protected:
    sycl::queue queue_;
    std::size_t length_;
};

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DFT_DESCRIPTOR_IMPL_HPP_

