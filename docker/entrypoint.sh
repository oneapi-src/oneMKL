#!/bin/bash

# save args
entrypoint_args=("$@")
# clear args
set --

PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\u@\h:\w\$'

#ls /opt/intel/oneapi/set
# Set oneAPI environment
source /opt/intel/oneapi/setvars.sh

dpcpp --version

# Set CUDA environment
#export PATH=/usr/local/cuda-10.1/bin:${PATH}

exec "${entrypoint_args[@]}"
