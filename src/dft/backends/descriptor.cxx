#include <iostream>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "mkl_version.h"

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/dft/types.hpp"

#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"
#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/dft/detail/mklcpu/onemkl_dft_mklcpu.hpp"
#include "mkl_dfti.h"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

template <precision prec, domain dom>
class descriptor_derived_impl : public oneapi::mkl::dft::detail::descriptor_impl {
public:
    descriptor_derived_impl(std::size_t length) : oneapi::mkl::dft::detail::descriptor_impl(length) {
        prec_ = prec;
        dom_ = dom;
    }

    descriptor_derived_impl(std::vector<std::int64_t> dimensions)
            : oneapi::mkl::dft::detail::descriptor_impl(dimensions) {
        prec_ = prec;
        dom_ = dom;
    }

    descriptor_derived_impl(const descriptor_derived_impl* other) : oneapi::mkl::dft::detail::descriptor_impl(*other) {
        std::cout << "special entry points copy const" << std::endl;
    }

    template<typename ...Types>
    void set_value(config_param param, Types... args) {
        printf("test... derived\n");
    }

    virtual oneapi::mkl::dft::detail::descriptor_impl* copy_state() override {
        return new descriptor_derived_impl(this);
    }

    virtual ~descriptor_derived_impl() override {
        std::cout << "descriptor_derived_impl descriptor" << std::endl;
        std::cout << values.bwd_scale << std::endl;
    }

private:
    DFTI_DESCRIPTOR_HANDLE hand;
};

// base constructor specialized
template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>(std::size_t length) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>(length);
}

template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>(std::size_t length) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>(length);
}

template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(std::size_t length) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(length);
}

template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>(std::size_t length) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>(length);
}

// vectorized constructor specialized
template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>(
    std::vector<std::int64_t> dimensions) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>(
        dimensions);
}

template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>(
    std::vector<std::int64_t> dimensions) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>(dimensions);
}

template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(
    std::vector<std::int64_t> dimensions) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(
        dimensions);
}

template <>
oneapi::mkl::dft::detail::descriptor_impl*
create_descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>(
    std::vector<std::int64_t> dimensions) {
    return new descriptor_derived_impl<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>(dimensions);
}

} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi
