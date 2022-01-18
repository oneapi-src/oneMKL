/*******************************************************************************
 * rocRAND back-end Copyright (c) 2021, The Regents of the University of
 * California, through Lawrence Berkeley National Laboratory (subject to receipt
 * of any required approvals from the U.S. Dept. of Energy). All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * (1) Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * (2) Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * (3) Neither the name of the University of California, Lawrence Berkeley
 * National Laboratory, U.S. Dept. of Energy nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * You are under no obligation whatsoever to provide any bug fixes, patches,
 * or upgrades to the features, functionality or performance of the source
 * code ("Enhancements") to anyone; however, if you choose to make your
 * Enhancements available either publicly, or directly to Lawrence Berkeley
 * National Laboratory, without imposing a separate written license agreement
 * for such Enhancements, then you hereby grant the following license: a
 * non-exclusive, royalty-free perpetual license to install, use, modify,
 * prepare derivative works, incorporate into other computer software,
 * distribute, and sublicense such enhancements or derivative works thereof,
 * in binary and source code form.
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
 * works, and perform publicly and display publicly, and to permit others to do
 * so.
 ******************************************************************************/

/**
 * @file curand_helper.cpp : contains the implementation of all the routines
 * for HIP backend
 */
#ifndef _MKL_RNG_ROCRAND_HELPER_HPP_
#define _MKL_RNG_ROCRAND_HELPER_HPP_
//#include <cuda.h>
#include <rocrand.h>

#include <complex>

#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace rocrand {

class rocrand_error : virtual public std::runtime_error {
protected:
    inline const char* rocrand_error_map(rocrand_status error) {
        switch (error) {
            case ROCRAND_STATUS_SUCCESS: return "ROCRAND_STATUS_SUCCESS";

            case ROCRAND_STATUS_NOT_CREATED: return "ROCRAND_STATUS_NOT_CREATED";

            case ROCRAND_STATUS_ALLOCATION_FAILED: return "ROCRAND_STATUS_ALLOCATION_FAILED";

            case ROCRAND_STATUS_TYPE_ERROR: return "ROCRAND_STATUS_TYPE_ERROR";

            case ROCRAND_STATUS_OUT_OF_RANGE: return "ROCRAND_STATUS_OUT_OF_RANGE";

            case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE: return "ROCRAND_STATUS_LENGTH_NOT_MULTIPLE";

            case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED";

            case ROCRAND_STATUS_LAUNCH_FAILURE: return "ROCRAND_STATUS_LAUNCH_FAILURE";

            case ROCRAND_STATUS_VERSION_MISMATCH: return "ROCRAND_STATUS_VERSION_MISMATCH";

            case ROCRAND_STATUS_INTERNAL_ERROR: return "ROCRAND_STATUS_INTERNAL_ERROR";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, rocrand_status).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit rocrand_error(std::string message, rocrand_status result)
            : std::runtime_error((message + std::string(rocrand_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~rocrand_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class cuda_error : virtual public std::runtime_error {
protected:
    inline const char* cuda_error_map(hipError_t result) {
        switch (result) {
            case HIP_SUCCESS: return "HIP_SUCCESS";

            case HIP_ERROR_NOT_INITIALIZED: return "HIP_ERROR_NOT_INITIALIZED";

            case hipErrorInvalidContext: return "HIP_ERROR_INVALID_CONTEXT";

            case hipErrorInvalidDevice: return "HIP_ERROR_INVALID_DEVICE";

            case HIP_ERROR_INVALID_VALUE: return "HIP_ERROR_INVALID_VALUE";

            case hipErrorRuntimeMemory: return "HIP_ERROR_OUT_OF_MEMORY";

            case HIP_ERROR_LAUNCH_OUT_OF_RESOURCES: return "HIP_ERROR_LAUNCH_OUT_OF_RESOURCES";

            default: return "<unknown>";
        }
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, CUresult).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit cuda_error(std::string message, hipError_t result)
            : std::runtime_error((message + std::string(cuda_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cuda_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define HIP_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                           \
    if (err != HIP_SUCCESS) {                                          \
        throw hip_error(std::string(#name) + std::string(" : "), err); \
    }

#define ROCRAND_CALL(func, status, ...)                                       \
    status = func(__VA_ARGS__);                                               \
    if (status != ROCRAND_STATUS_SUCCESS) {                                   \
        throw rocrand_error(std::string(#func) + std::string(" : "), status); \
    }

} // namespace rocrand
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_ROCRAND_HELPER_HPP_