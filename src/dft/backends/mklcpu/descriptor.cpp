#include <iostream>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "mkl_version.h"

#include "oneapi/mkl/types.hpp"

#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"
#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"

#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace mklcpu {

class descriptor_derived_impl : public oneapi::mkl::dft::detail::descriptor_impl {
public:
    descriptor_derived_impl(std::size_t length)
        : oneapi::mkl::dft::detail::descriptor_impl(length) {
        std::cout << "special entry points" << std::endl;
        DFTI_DESCRIPTOR_HANDLE hand = NULL;
    }

    descriptor_derived_impl(const descriptor_derived_impl* other)
        : oneapi::mkl::dft::detail::descriptor_impl(*other) {
        std::cout << "special entry points copy const" << std::endl;
    }

    virtual oneapi::mkl::dft::detail::descriptor_impl* copy_state() override {
        return new descriptor_derived_impl(this);
    }

    virtual ~descriptor_derived_impl() override {
        std::cout << "descriptor_derived_impl descriptor" << std::endl;
    }
private:
};

oneapi::mkl::dft::detail::descriptor_impl* create_descriptor(std::size_t length) {
    return new descriptor_derived_impl(length);
}



} // namespace mklcpu
} // namespace dft
} // namespace mkl
} // namespace oneapi
