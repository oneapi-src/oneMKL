//
// generated file
//

#include "blas/function_table.hpp"

#include "oneapi/mkl/blas/detail/syclblas/onemkl_blas_syclblas.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMKL_EXPORT blas_function_table_t mkl_blas_table = {
    WRAPPER_VERSION,
#define BACKEND syclblas
#define MAJOR   column_major
#include "../backend_wrappers.cxx"
#undef MAJOR
#define MAJOR row_major
#include "../backend_wrappers.cxx"
#undef MAJOR
#undef BACKEND
};
