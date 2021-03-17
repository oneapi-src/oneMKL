/*********************************************************************************
* Intel Math Kernel Library (oneMKL) Copyright (c) 2021, The Regents of
* the University of California, through Lawrence Berkeley National
* Laboratory (subject to receipt of any required approvals from the U.S.
* Dept. of Energy). All rights reserved.
* 
* If you have questions about your rights to use or distribute this software,
* please contact Berkeley Lab's Intellectual Property Office at
* IPO@lbl.gov.
* 
* NOTICE.  This Software was developed under funding from the U.S. Department
* of Energy and the U.S. Government consequently retains certain rights.  As
* such, the U.S. Government has been granted for itself and others acting on
* its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
* Software to reproduce, distribute copies to the public, prepare derivative 
* works, and perform publicly and display publicly, and to permit others to do so.
*********************************************************************************/

#ifndef _ONEMKL_RNG_CURAND_HPP_
#define _ONEMKL_RNG_CURAND_HPP_

#include <cstdint>
#include <CL/sycl.hpp>

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/rng/detail/engine_impl.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace curand {

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(cl::sycl::queue queue,
                                                                          std::uint64_t seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_philox4x32x10(
    cl::sycl::queue queue, std::initializer_list<std::uint64_t> seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_mrg32k3a(cl::sycl::queue queue,
                                                                     std::uint32_t seed);

ONEMKL_EXPORT oneapi::mkl::rng::detail::engine_impl* create_mrg32k3a(
    cl::sycl::queue queue, std::initializer_list<std::uint32_t> seed);

} // namespace curand
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif //_ONEMKL_RNG_CURAND_HPP_
