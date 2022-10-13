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
    descriptor_impl();
    ~descriptor_impl() {}

protected:
    sycl::queue queue_;
    void* handle_;
};

template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
oneapi::mkl::dft::detail::descriptor_impl* create_commit(oneapi::mkl::device libkey, sycl::queue queue) {
    return new descriptor_impl();
}

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DFT_DESCRIPTOR_IMPL_HPP_

