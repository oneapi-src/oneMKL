#!/bin/bash

# save args
entrypoint_args=("$@")
# clear args
#set --

PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\u@\h:\w\$'

# Set oneAPI environment
whoami
source /opt/intel/oneapi/setvars.sh
dpcpp --version

# Start build
cd BUILD
conan --version
conan install .. -pr inteldpcpp_lnx --build missing
# NETLIB Package for LAPACK is no longer available on the official Conan repo.
# TODO: Fix Reference BLAS and LAPACK in Conan builds.

# Set CUDA environment
#export PATH=/usr/local/cuda-10.1/bin:${PATH}

exec "${entrypoint_args[@]}"
