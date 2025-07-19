if USE_CUDA
lib_LTLIBRARIES = librinhash_cuda.la
librinhash_cuda_la_SOURCES = \
    rinhash.cu \
    blake3_device.cuh \
    argon2d_device.cuh \
    sha3-256.cu \
    rinhash_device.cuh

librinhash_cuda_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS)
librinhash_cuda_la_LDFLAGS = $(CUDA_LIBS)

# CUDA compilation rules
.cu.o:
	$(NVCC) $(CUDA_CFLAGS) -c $< -o $@
	
.cu.lo:
	$(LIBTOOL) --mode=compile $(NVCC) $(CUDA_CFLAGS) -c $< -o $@
endif
