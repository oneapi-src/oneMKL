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

#include "rng/function_table.hpp"
#include "oneapi/mkl/rng/detail/curand/onemkl_rng_curand.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMKL_EXPORT rng_function_table_t mkl_rng_table = {
    WRAPPER_VERSION, oneapi::mkl::rng::curand::create_philox4x32x10,
    oneapi::mkl::rng::curand::create_philox4x32x10, oneapi::mkl::rng::curand::create_mrg32k3a,
    oneapi::mkl::rng::curand::create_mrg32k3a
};
