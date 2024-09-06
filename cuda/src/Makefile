#
#  Makefile to build libcuda-crypt.so
#   From sources in /cuda-crypt and /ed25519-verify
#
#   nvcc inserts conflicting driver symbols into a static lib (.so)
#   so more than one .so cannot be linked into a single program.
#   Shared objects with device kernels also did not seem to work--
#   it can build, but the kernel launch is not successful. (with cuda 9.2)
#   Hence, build ed25519 ecdsa verify and chacha encryption device
#   code into a single binary.

V=release
include gpu-common.mk

# CHACHA_TEST_BIN=cuda_chacha_test
# ECC_TEST_BIN=cuda_ed25519_verify
ECC_VANITY=cuda-vanity
# LIB=cuda-crypt
# LIB=cuda-ed25519-vanity

all: $V/lib-$(ECC_VANITY).so $V/$(ECC_VANITY)

CUDA_HEADER_DIR:=cuda-headers
# CUDA_SHA256_DIR:=cuda-sha256
CUDA_ECC_DIR:=cuda-ecc-ed25519

CFLAGS+=-DENDIAN_NEUTRAL -DLTC_NO_ASM -I$(CUDA_HEADER_DIR)

$V/ecc_vanity.o: $(addprefix $(CUDA_ECC_DIR)/,vanity.cu sha512.h fe.h ge.h ../$(CUDA_HEADER_DIR)/gpu_common.h)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

$V/lib-$(ECC_VANITY).so: $(addprefix $V/,ecc_vanity.o)
	$(NVCC) -Xcompiler "-fPIC" --shared --output-file $@ $^

$V/$(ECC_VANITY): $(addprefix $V/,ecc_vanity.o)
	$(NVCC) $(CFLAGS) -L$V $< -o $@

.PHONY:clean
clean:
	rm -rf $V

# test: $V/$(ECC_TEST_BIN) $V/$(CHACHA_TEST_BIN)
# 	cd $(V) && ./$(CHACHA_TEST_BIN) 64 \
# 	cd $(V) && ./$(ECC_TEST_BIN) 1 1 1 1 1 1
# 	cd $(V) && ./$(ECC_TEST_BIN) 64 1 1 1 1 0
# 	cd $(V) && ./$(ECC_TEST_BIN) 100201 1 1 4 10 1
