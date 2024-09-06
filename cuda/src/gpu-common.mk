NVCC:=nvcc
# ===============================================
# For generation on CUDA 11.7 for maximum
# compatibility with V100 and T4 Turing =
# Datacenter cards, but also support newer RTX
# 3080, and Drive AGX Orin
# ===============================================
GPU_CFLAGS:=-arch=sm_52 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_86,code=sm_86 \
-gencode=arch=compute_87,code=sm_87 \
-gencode=arch=compute_86,code=compute_86
# ===============================================
# For generation on CUDA 11.4 for best performance
# with RTX 3080 cards:
# ===============================================
# GPU_CFLAGS:=-arch=sm_80 \
# -gencode=arch=compute_80,code=sm_80 \
# -gencode=arch=compute_86,code=sm_86 \
# -gencode=arch=compute_87,code=sm_87 \
# -gencode=arch=compute_86,code=compute_86
# ===============================================
# For generation on CUDA 12 for best performance
# with GeForce RTX 4080:
# ===============================================
# GPU_CFLAGS:=-arch=sm_89 \ 
# -gencode=arch=compute_89,code=sm_89 \
# -gencode=arch=compute_89,code=compute_89

CFLAGS_release:=--ptxas-options=-v $(GPU_CFLAGS) -O3 -Xcompiler "-Wall -Werror -fPIC -Wno-strict-aliasing"
CFLAGS_debug:=$(CFLAGS_release) -g
CFLAGS:=$(CFLAGS_$V)
