#ifndef _ONEMKL_DFT_COMMIT_IMPL_HPP_
#define _ONEMKL_DFT_COMMIT_IMPL_HPP_

#include <cstdint>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"
#include "oneapi/mkl/dft/types.hpp"

#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

class commit_impl {
public:
    commit_impl(sycl::queue queue) : queue_(queue), handle(nullptr) {}

    commit_impl(const commit_impl& other) : queue_(other.queue_), handle(other.handle) {}

    virtual commit_impl* copy_state() = 0;

    virtual ~commit_impl() {}

    sycl::queue& get_queue() {
        return queue_;
    }

protected:
    bool status;
    sycl::queue queue_;
    void* handle;
};


} // namespace detail
} // namespace dft
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_DFT_COMMIT_IMPL_HPP_

