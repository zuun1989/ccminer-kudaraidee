/**
 * Simplified groestl512 big perm code
 * tpruvot - 2017
 */

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <cuda_helper.h>
#include <cuda_texture_types.h>
#define __byte_perm(a,b,c) (a)
#define tex1D(t, n) (t[n])
#endif

// todo: merge with cuda_quark_groestl512_sm20.cu (used for groestl512-80)

#if __CUDA_ARCH__ < 300 || defined(_DEBUG)

#ifndef SPH_C32
#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))
#endif

#ifndef PC32up
#define PC32up(j, r)   ((uint32_t)((j) + (r)))
#define PC32dn(j, r)   0
#define QC32up(j, r)   0xFFFFFFFF
#define QC32dn(j, r)   (((uint32_t)(r) << 24) ^ SPH_T32(~((uint32_t)(j) << 24)))
#endif

#define tT0up(x) tex1D<unsigned int>(t0up1, x)
#define tT0dn(x) tex1D<unsigned int>(t0dn1, x)
#define tT1up(x) tex1D<unsigned int>(t1up1, x)
#define tT1dn(x) tex1D<unsigned int>(t1dn1, x)
#define tT2up(x) tex1D<unsigned int>(t2up1, x)
#define tT2dn(x) tex1D<unsigned int>(t2dn1, x)
#define tT3up(x) tex1D<unsigned int>(t3up1, x)
#define tT3dn(x) tex1D<unsigned int>(t3dn1, x)

#undef B32_0
#define B32_0(x) ((x) & 0xFFu)
#define B32_1(x) (((x) >> 8) & 0xFFu)
#define B32_2(x) (((x) >> 16) & 0xFFu)
#define B32_3(x) (((x) >> 24) & 0xFFu)

__device__ __forceinline__
static void tex_groestl512_perm_P(uint32_t *a)
{
	#pragma unroll 1
	for(int r=0; r<14; r++)
	{
		uint32_t t[32];

		#pragma unroll 16
		for (int k=0; k<16; k++)
			a[(k*2)+0] ^= PC32up(k<< 4, r);

		#pragma unroll 16
		for(int k=0; k<32; k+=2)
        {
            uint32_t t0_0 = B32_0(a[(k    ) & 0x1f]), t9_0  = B32_0(a[(k +  9) & 0x1f]);
            uint32_t t2_1 = B32_1(a[(k + 2) & 0x1f]), t11_1 = B32_1(a[(k + 11) & 0x1f]);
            uint32_t t4_2 = B32_2(a[(k + 4) & 0x1f]), t13_2 = B32_2(a[(k + 13) & 0x1f]);
            uint32_t t6_3 = B32_3(a[(k + 6) & 0x1f]), t23_3 = B32_3(a[(k + 23) & 0x1f]);

            t[k + 0] = __ldg(&t0_0) ^ __ldg(&t2_1) ^ __ldg(&t4_2) ^ __ldg(&t6_3) ^
                        __ldg(&t9_0) ^ __ldg(&t11_1) ^ __ldg(&t13_2) ^ __ldg(&t23_3);

            t[k + 1] = __ldg(&t0_0) ^ __ldg(&t2_1) ^ __ldg(&t4_2) ^ __ldg(&t6_3) ^
                        __ldg(&t9_0) ^ __ldg(&t11_1) ^ __ldg(&t13_2) ^ __ldg(&t23_3);
        }

		#pragma unroll 32
		for(int k=0; k<32; k++)
			a[k] = t[k];
	}
}

__device__ __forceinline__
static void tex_groestl512_perm_Q(uint32_t *a)
{
	#pragma unroll 1
	for(int r=0; r<14; r++)
	{
		uint32_t t[32];

		#pragma unroll 16
		for (int k=0; k<16; k++) {
			a[(k*2)+0] ^= QC32up(k<< 4, r);
			a[(k*2)+1] ^= QC32dn(k<< 4, r);
		}

		#pragma unroll 16
		for(int k=0; k<32; k+=2)
        {
            uint32_t t2_0  = B32_0(a[(k +  2) & 0x1f]), t1_0  = B32_0(a[(k +  1) & 0x1f]);
            uint32_t t6_1  = B32_1(a[(k +  6) & 0x1f]), t5_1  = B32_1(a[(k +  5) & 0x1f]);
            uint32_t t10_2 = B32_2(a[(k + 10) & 0x1f]), t9_2  = B32_2(a[(k +  9) & 0x1f]);
            uint32_t t22_3 = B32_3(a[(k + 22) & 0x1f]), t13_3 = B32_3(a[(k + 13) & 0x1f]);

            t[k + 0] = __ldg(&t2_0) ^ __ldg(&t6_1) ^ __ldg(&t10_2) ^ __ldg(&t22_3) ^
                        __ldg(&t1_0) ^ __ldg(&t5_1) ^ __ldg(&t9_2) ^ __ldg(&t13_3);

            t[k + 1] = __ldg(&t2_0) ^ __ldg(&t6_1) ^ __ldg(&t10_2) ^ __ldg(&t22_3) ^
                        __ldg(&t1_0) ^ __ldg(&t5_1) ^ __ldg(&t9_2) ^ __ldg(&t13_3);
        }

		#pragma unroll 32
		for(int k=0; k<32; k++)
			a[k] = t[k];
	}
}

#endif
