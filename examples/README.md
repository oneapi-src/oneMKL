# oneAPI Math Kernel Library (oneMKL) Interfaces Examples 
oneAPI Math Kernel Library (oneMKL) Interfaces offers examples with the following routines: 
- blas: level3/gemm_usm  
- rng: uniform_usm  
- lapack: getrs_usm
- dft: complex_fwd_buffer, real_fwd_usm
- sparse_blas: sparse_gemv_usm

Each routine has one run-time dispatching example and one compile-time dispatching example (which uses both mklcpu and cuda backends), located in `example/<$domain>/run_time_dispatching` and `example/<$domain>/compile_time_dispatching` subfolders, respectively.

To build examples, use cmake build option `-DBUILD_EXAMPLES=true`.  
Compile_time_dispatching will be built if `-DBUILD_EXAMPLES=true` and cuda backend is enabled, because the compile-time dispatching example runs on both mklcpu and cuda backends.
Run_time_dispatching will be built if `-DBUILD_EXAMPLES=true` and `-DBUILD_SHARED_LIBS=true`.
All DFT examples require the mklgpu backend to be enabled.

The example executable naming convention follows `example_<$domain>_<$routine>_<$backend>` for compile-time dispatching examples 
  or `example_<$domain>_<$routine>` for run-time dispatching examples. 
  E.g. `example_blas_gemm_usm_mklcpu_cublas `  `example_blas_gemm_usm`

## Running examples (blas)
  
## blas

Below are showcases of how to run examples with different backends using the BLAS domain as an illustration.

Run-time dispatching examples with mklcpu backend
```
$ export ONEAPI_DEVICE_SELECTOR="opencl:cpu"
$ ./bin/example_blas_gemm_usm
```
Run-time dispatching examples with mklgpu backend
```
$ export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
$ ./bin/example_blas_gemm_usm
```
Compile-time dispatching example with both mklcpu and cublas backend

(Note that the mklcpu and cublas result matrices have a small difference. This is expected due to precision limitation of `float`)
```
./bin/example_blas_gemm_usm_mklcpu_cublas
```
