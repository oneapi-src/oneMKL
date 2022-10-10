#ifndef _ONEMKL_DFT_LOADER_HPP_
#define _ONEMKL_DFT_LOADER_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"

namespace oneapi {
namespace mkl {
namespace dft {

namespace mklcpu {

ONEMKL_EXPORT oneapi::mkl::dft::detail::descriptor_impl* create_descriptor(std::size_t length);

} // namespace mklcpu

namespace mklgpu {

ONEMKL_EXPORT oneapi::mkl::dft::detail::descriptor_impl* create_descriptor(std::size_t length);

} // namespace mklgpu

} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DFT_LOADER_HPP_
