TODO: update description for users before merging

#### Examples are added for routines in each domain:  
- blas routine: gemm_usm  
- rng routine: uniform_usm  
- lapack routine: getrf followed by getrs with usm  

Each routine has a run-time dispatching example and three compile-time idspatching examples (for mklcpu, mklgpu, and cuda backeds), located in `example/<$domain>/run_time_dispatching` and `example/<$domain>/compile_time_dispatching` subfolders, respectively.

One new cmake build option `-DBUILD_EXAMPLES` is added.  
Compile_time_dispatching will always be built if `-DBUILD_EXAMPLES=true`.   
Run_time_dispatching will be build if `-DBUILD_EXAMPLES=true` and `-DBUILD_SHARED_LIBS=true`
  
Example executables are inside `bin/`. The executable naming convention follows `example_<$domain>_<$routine>_<$backend>` for compile-time dispatching examples 
  or `example_<$domain>_<$routine>` for run-time dispatching examples. 
  E.g. `example_blas_gemm_usm_mklcpu `  `example_blas_gemm_usm`

## Test outputs (blas, rng, lapack)
  
### 1. blas: gemm with USM
  
###   blas compile time dispatching examples (mklcpu backend on jflmkl109)
![blas_ct_mklcpu](https://user-images.githubusercontent.com/46687831/163023906-8f033920-f345-4016-bfdd-9d5363927b09.JPG)
  
  
###   blas compile time dispatching examples (mklgpu backend on jflmkl109)
![blas_ct_mklgpu](https://user-images.githubusercontent.com/46687831/163023920-c24ccc90-7f22-4804-9f8e-ba316d638ebf.JPG)
  
  
###   blas compile time dispatching examples (mklcpu and cublas backend on jflmkl125)
![blas_ct_nvidia](https://user-images.githubusercontent.com/46687831/163025370-6a266c2c-4f96-4843-a305-c810a352a6eb.JPG)

  
### blas run time dispatching examples on jflmkl111
  ![blas_rt_111_(1)](https://user-images.githubusercontent.com/46687831/163025751-59f53222-cd1c-4b6d-aa42-5020d51fa4ce.JPG)
  
  
 ### blas run time dispatching examples on jflmkl125
  ![blas_rt_125](https://user-images.githubusercontent.com/46687831/163025094-112c419a-4b1e-4e69-9626-3ad47065d920.JPG)

  
  
## 2. rng: uniform with USM
  
### rng compile time dispatching examples on jflmkl125
  ![rng_ct_125](https://user-images.githubusercontent.com/46687831/163026432-e28a320b-4415-4688-9a7a-f03f72770f60.JPG)
  
### rng compile time dispatching examples on jflmkl111
![rng_ct_mkl](https://user-images.githubusercontent.com/46687831/163026471-9dd9e85e-f456-4294-8e62-ea19a95e7487.JPG)

### rng run time dispatching examples on jflmkl109
![rng_rt_111](https://user-images.githubusercontent.com/46687831/163026484-a434d377-18cf-4374-b226-052b30082cf0.JPG)
  
### rng run time dispatching examples on jflmkl125
![rng_rt_125](https://user-images.githubusercontent.com/46687831/163026495-2c973f85-0173-4fbb-b55a-7658ebff4748.JPG)
 
## 3. lapack: getrf followed by getrs with USM

### lapack compile time dispatching examples on jflmkl109
  ![lapack_ct_mklcpugpu](https://user-images.githubusercontent.com/46687831/163026270-a581002f-e26d-4e03-b21b-5e3f8b446109.JPG)

### lapack run time dispatching examples jflmkl109**
  ![lapack_rt_111](https://user-images.githubusercontent.com/46687831/163026288-b0770197-9009-4b0f-9301-9171d0734187.JPG)




