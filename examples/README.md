# oneAPI Math Kernel Library (oneMKL) Interfaces Examples 
oneAPI Math Kernel Library (oneMKL) Interfaces offers examples with the following routines: 
- blas: level3/gemm_usm  
- rng: uniform_usm  
- lapack: getrs_usm

Each routine has a run-time dispatching example and three compile-time dispatching examples (for mklcpu, mklgpu, and cuda backends), located in `example/<$domain>/run_time_dispatching` and `example/<$domain>/compile_time_dispatching` subfolders, respectively.

To build examples, use cmake build option `-DBUILD_EXAMPLES=true`.  
Compile_time_dispatching will always be built if `-DBUILD_EXAMPLES=true`.   
Run_time_dispatching will be build if `-DBUILD_EXAMPLES=true` and `-DBUILD_SHARED_LIBS=true`

The example executable naming convention follows `example_<$domain>_<$routine>_<$backend>` for compile-time dispatching examples 
  or `example_<$domain>_<$routine>` for run-time dispatching examples. 
  E.g. `example_blas_gemm_usm_mklcpu `  `example_blas_gemm_usm`

## Test outputs (blas, rng, lapack)
  
## blas
Compile-time dispatching examples with mklcpu backend
```
$ ./bin/example_blas_gemm_usm_mklcpu

########################################################################
# General Matrix-Matrix Multiplication using Unified Shared Memory Example:
#
# C = alpha * A * B + beta * C
#
# where A, B and C are general dense matrices and alpha, beta are
# floating point type precision scalars.
#
# Using apis:
#   gemm
#
# Supported floating point type precisions:
#   float
#
########################################################################

Running BLAS gemm usm example on CPU device. Device name is: Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz.
        Running with single precision real data type:
GEMM_MKL_CPU

                GEMM parameters:
                        transA = trans, transB = nontrans
                        m = 45, n = 98, k = 67
                        lda = 103, ldB = 105, ldC = 106
                        alpha = 2, beta = 3

                Outputting 2x2 block of A,B,C matrices:

                        A = [ 0.340188, 0.260249, ...
                            [ -0.105617, 0.0125354, ...
                            [ ...


                        B = [ -0.326421, -0.192968, ...
                            [ 0.363891, 0.251295, ...
                            [ ...


                        C = [ 0.00698781, 0.525862, ...
                            [ 0.585167, 1.59017, ...
                            [ ...
```

Run-time dispatching examples with mklcpu backend
```
$ export SYCL_DEVICE_FILTER=cpu
$ ./bin/example_blas_gemm_usm

########################################################################
# General Matrix-Matrix Multiplication using Unified Shared Memory Example:
#
# C = alpha * A * B + beta * C
#
# where A, B and C are general dense matrices and alpha, beta are
# floating point type precision scalars.
#
# Using apis:
#   gemm
#
# Supported floating point type precisions:
#   float
#
########################################################################

Running BLAS gemm usm example on CPU device. Device name is: Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz.
        Running with single precision real data type:
Runtime compilation, backend not specified

                GEMM parameters:
                        transA = trans, transB = nontrans
                        m = 45, n = 98, k = 67
                        lda = 103, ldB = 105, ldC = 106
                        alpha = 2, beta = 3

                Outputting 2x2 block of A,B,C matrices:

                        A = [ 0.340188, 0.260249, ...
                            [ -0.105617, 0.0125354, ...
                            [ ...


                        B = [ -0.326421, -0.192968, ...
                            [ 0.363891, 0.251295, ...
                            [ ...


                        C = [ 0.00698781, 0.525862, ...
                            [ 0.585167, 1.59017, ...
                            [ ...
```

Run-time dispatching examples with cublas backend
```
$ export SYCL_DEVICE_FILTER=cuda:gpu
$ ./bin/example_blas_gemm_usm

########################################################################
# General Matrix-Matrix Multiplication using Unified Shared Memory Example:
#
# C = alpha * A * B + beta * C
#
# where A, B and C are general dense matrices and alpha, beta are
# floating point type precision scalars.
#
# Using apis:
#   gemm
#
# Supported floating point type precisions:
#   float
#
########################################################################

Running BLAS gemm usm example on GPU device. Device name is: TITAN RTX.
        Running with single precision real data type:
Runtime compilation, backend not specified

                GEMM parameters:
                        transA = trans, transB = nontrans
                        m = 45, n = 98, k = 67
                        lda = 103, ldB = 105, ldC = 106
                        alpha = 2, beta = 3

                Outputting 2x2 block of A,B,C matrices:

                        A = [ 0.340188, 0.260249, ...
                            [ -0.105617, 0.0125354, ...
                            [ ...


                        B = [ -0.326421, -0.192968, ...
                            [ 0.363891, 0.251295, ...
                            [ ...


                        C = [ 0.00698793, 0.525862, ...
                            [ 0.585168, 1.59017, ...
                            [ ...
```
 
## lapack 
Run-time dispatching example with mklgpu backend:
```
$ export SYCL_DEVICE_FILTER=gpu
$ ./bin/example_lapack_getrs_usm
########################################################################
# LU Factorization and Solve Example:
#
# Computes LU Factorization A = P * L * U
# and uses it to solve for X in a system of linear equations:
#   AX = B
# where A is a general dense matrix and B is a matrix whose columns
# are the right-hand sides for the systems of equations.
#
# Using apis:
#   getrf and getrs
#
# Supported floating point type precisions:
#   float
#
########################################################################
Running LAPACK getrs example on GPU device. Device name is: Intel(R) Iris(R) Pro Graphics 580 [0x193b].
  Running with single precision real data type:
getrs ran OK
```

## rng
Run-time dispatching example with mklcpu backend:
```
$ export SYCL_DEVICE_FILTER=cpu
$ ./bin/example_rng_uniform_usm
########################################################################
# Generate uniformly distributed random numbers with philox4x32x10
# generator example:
#
# Using APIs:
#   default_engine uniform
#  
# Supported precisions:
#   float
# 
########################################################################

Running RNG uniform usm example on CPU device.
Device name is: Intel(R) Core(TM) i7-6770HQ CPU @ 2.60GHz
        Running with single precision real data type:

                generation parameters:
                        seed = 777, a = 0, b = 10

                Output of generator:
first 10 numbers of 1000:
8.52971 1.76033 6.04753 3.68079 9.04039 2.61014 3.75788 3.94859 7.93444 8.60436
Success: sample moments (mean=5.01785, variance=8.4075) agree with theory (mean=5, variance=8.33333)
PASSED
```


