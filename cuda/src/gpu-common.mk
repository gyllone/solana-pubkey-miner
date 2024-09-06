NVCC:=nvcc
# GPU_PTX_ARCH:=compute_90
# GPU_ARCHS?=sm_90
GPU_PTX_ARCH:=compute_52
GPU_ARCHS?=sm_52,sm_70,sm_90,compute_52
GPU_CFLAGS:=--gpu-code=$(GPU_ARCHS) --gpu-architecture=$(GPU_PTX_ARCH)
CFLAGS_release:=--ptxas-options=-v $(GPU_CFLAGS) -O3 -Xcompiler "-Wall -Werror -fPIC -Wno-strict-aliasing"
CFLAGS_debug:=$(CFLAGS_release) -g
CFLAGS:=$(CFLAGS_$V)
