#!/bin/bash

# export OMP_TARGET_OFFLOAD='DISABLED'
# export LIBOMPTARGET_PLUGIN_PROFILE=T
# export LIBOMPTARGET_DEBUG=1

export MKL_VERBOSE=2
export MKL_FFT_TRACE=2
export MKL_FFT_LLK_TRACE=2

# export CreateMultipleSubDevices=1
# export EnableWalkerPartition=0
export ZE_AFFINITY_MASK=0.0
# export EnableImplicitScaling=1
driver=/nfs/site/proj/mkl/mirror/NN/tools/drivers/gpu/lnx/unpacked/2022.0.0.22599
compiler_path=/nfs/pdx/disks/mkl_mirror/NN/tools/dpc++/20230129_rls/lnx

path_to_mkl="/nfs/pdx/disks/mkl_project/anantsri/worktree/large_prime/src/R_dft_develop_GPU_DBG_20221202_rls_20210722_000000_gnx32e/"
#rls="R_dft_offload_perf_GPU_DBG_20220302_rls_20210722_000000_gnx32e"
#release_dir="${path_to_mkl}/src/${rls}"
#release_dir="/nfs/pdx/disks/mkl_project/anantsri/build/__release_lnx/"
release_dir="/nfs/pdx/disks/mkl_project/anantsri/libraries.performance.math.mkl/src/R_dft_reset_bkd_GPU_DBG_20230210_rls_20210722_000000_gnx32e/"
# release_dir="/nfs/pdx/disks/mkl_project/anantsri/temp/R__develop04-28-2022-develop_GPU_20220302_rls_20210722_000000_gnx32e/"

COMPILER_VERSION=20230303_rls
source /nfs/site/proj/mkl/mirror/NN/tools/dpc++/${COMPILER_VERSION}/lnx/setvars.sh
export PATH=${driver}/bin:$PATH
export LD_LIBRARY_PATH=/nfs/site/home/anantsri/devel/oneMKL_anantsri/build/lib:${driver}/lib/x86_64-linux-gnu:${driver}/lib/x86_64-linux-gnu/intel-opencl:${compiler_path}/compiler/latest/linux/lib:${compiler_path}/compiler/latest/linux/lib/x64:${compiler_path}/compiler/latest/linux/lib/oclfpga/host/linux64/lib:${compiler_path}/compiler/latest/linux/compiler/lib/intel64_lin:/nfs/pdx/disks/mkl_mirror/NN/tools/tbb/lnx/2021.6-gold/tbb/env/../lib/intel64/gcc4.8:${LD_LIBRARY_PATH}

sycl-ls

export DEVICE=cpu
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
./$1
