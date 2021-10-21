/**
 * Lyra2 (v1) cuda implementation improved by fancyIX
 * fancyIX@github 2021
 */

#include <stdio.h>
#include <memory.h>

#define TPB52 32

#include "cuda_lyra2_sm2.cuh"
#include "cuda_lyra2_sm5.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 520
#endif

#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ > 500

#include "cuda_lyra2_vectors.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
__device__ uint32_t __shfl(uint32_t a, uint32_t b, uint32_t c);
#endif

#define Nrow 8
#define Ncol 8
#define memshift 3

#define BUF_COUNT 0

__device__ uint2 *DMatrix;

__device__ __forceinline__ void LD4S(uint2 res[3], const int row, const int col, const int thread, const int threads)
{
#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*blockDim.x + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = *(DMatrix + d0 + j * threads * blockDim.x);
#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
#else
	if (row < BUF_COUNT)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = *(DMatrix + d0 + j * threads * blockDim.x);
	}
	else
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
	}
#endif
}

__device__ __forceinline__ void ST4S(const int row, const int col, const uint2 data[3], const int thread, const int threads)
{
#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*blockDim.x + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		*(DMatrix + d0 + j * threads * blockDim.x) = data[j];

#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];

#else
	if (row < BUF_COUNT)
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + d0 + j * threads * blockDim.x) = data[j];
	}
	else
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];
	}
#endif
}

#if __CUDA_ARCH__ >= 300
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	return __shfl(a, b, c);
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	a1 = WarpShuffle(a1, b1, c);
	a2 = WarpShuffle(a2, b2, c);
	a3 = WarpShuffle(a3, b3, c);
}

#else
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;
	uint32_t *_ptr = (uint32_t*)shared_mem;

	__threadfence_block();
	uint32_t buf = _ptr[thread];

	_ptr[thread] = a;
	__threadfence_block();
	uint32_t result = _ptr[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	_ptr[thread] = buf;

	__threadfence_block();
	return result;
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a;
	__threadfence_block();
	uint2 result = shared_mem[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;

	__threadfence_block();
	return result;
}

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a1;
	__threadfence_block();
	a1 = shared_mem[(thread&~(c - 1)) + (b1&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a2;
	__threadfence_block();
	a2 = shared_mem[(thread&~(c - 1)) + (b2&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a3;
	__threadfence_block();
	a3 = shared_mem[(thread&~(c - 1)) + (b3&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;
	__threadfence_block();
}

#endif

#if __CUDA_ARCH__ > 500 || !defined(__CUDA_ARCH)
static __device__ __forceinline__
void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; uint2 tmp = d; d.y = a.x ^ tmp.x; d.x = a.y ^ tmp.y;
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
}
#endif

__device__ __forceinline__ void round_lyra(uint2 s[4])
{
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

static __device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__ __forceinline__
void reduceDuplex(uint2 state[4], uint32_t thread, const uint32_t threads)
{
	uint2 state1[3];

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < Nrow; i++)
	{
		ST4S(0, Ncol - i - 1, state, thread, threads);

		round_lyra(state);
	}

	#pragma unroll 4
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, 0, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];
		ST4S(1, Ncol - i - 1, state1, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	#pragma unroll 1
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, Ncol - i - 1, state1, thread, threads);

		// simultaneously receive data from preceding thread and send data to following thread
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	for (int i = 0; i < Nrow; i++)
	{
		uint2 state1[3], state2[3];

		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		// simultaneously receive data from preceding thread and send data to following thread
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);

		LD4S(state1, rowOut, i, thread, threads);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, i, state1, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowt_8(const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3], last[3];

	LD4S(state1, 2, 0, thread, threads);
	LD4S(last, rowInOut, 0, thread, threads);

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	// simultaneously receive data from preceding thread and send data to following thread
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	} else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 5)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{
		LD4S(state1, 2, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__constant__ uint2x4 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint2x4 state[4];

		state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state); //because 12 is not enough

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__
__launch_bounds__(TPB52, 1)
void lyra2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);

		reduceDuplex(state, thread, threads);
		reduceDuplexRowSetup(1, 0, 2, state, thread, threads);
		reduceDuplexRowSetup(2, 1, 3, state, thread, threads);
		reduceDuplexRowSetup(3, 0, 4, state, thread, threads);
		reduceDuplexRowSetup(4, 3, 5, state, thread, threads);
		reduceDuplexRowSetup(5, 2, 6, state, thread, threads);
		reduceDuplexRowSetup(6, 1, 7, state, thread, threads);

		uint32_t rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(7, rowa, 0, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(0, rowa, 3, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(3, rowa, 6, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(6, rowa, 1, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(1, rowa, 4, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(4, rowa, 7, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(7, rowa, 2, state, thread, threads);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt_8(rowa, state, thread, threads);

		DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		g_hash[thread + threads * 0] = state[0].x;
		g_hash[thread + threads * 1] = state[0].y;
		g_hash[thread + threads * 2] = state[0].z;
		g_hash[thread + threads * 3] = state[0].w;

	} //thread
}
#else
#if __CUDA_ARCH__ < 500

/* for unsupported SM arch */
__device__ void* DMatrix;
#endif
__global__ void lyra2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
__global__ void lyra2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash) {}
__global__ void lyra2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 320
// ============================ defines ========================
#define ADD32_DPP(a, b) \
	asm(" add.cc.u32  %0, %0, %2;\n\t" \
		" addc.u32 %1, 0, 0;\n\t" \
		" and.b32 %1, %1, %3;\n\t" \
		" shfl.sync.idx.b32  %1, %1, %4, 0x181F, 0xffffffff;\n\t" \
		" add.u32 %0, %0, %1;" \
		: "+r"(a), "+r"(zero): "r"(b), "r"(~player), "r"(warp_local + 4));
	

#define SWAP32_DPP(s) \
    ss = s; \
	{ \
		  asm(" shfl.sync.idx.b32  %0, %1, %2, 0x181F, 0xffffffff;\n\t" \
		      : "=r"(s) : "r"(ss), "r"(warp_local + 4)); \
	}

#define ROTR64_24_DPP(s) \
    ss = s; \
	{ \
		asm(" shfl.sync.idx.b32  %0, %0, %2, 0x181F, 0xffffffff;\n\t" \
			" shf.r.clamp.b32  %1, %1, %0, 24;" \
			: "+r"(ss), "+r"(s) : "r"(warp_local + 4)); \
	}

#define ROTR64_16_DPP(s) \
    ss = s; \
	{ \
		asm(" shfl.sync.idx.b32  %0, %0, %2, 0x181F, 0xffffffff;\n\t" \
			" shf.r.clamp.b32  %1, %1, %0, 16;" \
			: "+r"(ss), "+r"(s) : "r"(warp_local + 4)); \
	}

#define ROTR64_63_DPP(s) \
    ss = s; \
	{ \
		asm(" shfl.sync.idx.b32  %0, %0, %2, 0x181F, 0xffffffff;\n\t" \
			" shf.r.clamp.b32  %1, %0, %1, 31;" \
			: "+r"(ss), "+r"(s) : "r"(warp_local + 4)); \
	}

// Usually just #define G(a,b,c,d)...; I have no time to read the Lyra paper
// but that looks like some kind of block cipher I guess.
#define cipher_G_macro(s) \
    ADD32_DPP(s[0], s[1]); s[3] ^= s[0]; SWAP32_DPP(s[3]); \
    ADD32_DPP(s[2], s[3]); s[1] ^= s[2]; ROTR64_24_DPP(s[1]); \
    ADD32_DPP(s[0], s[1]); s[3] ^= s[0]; ROTR64_16_DPP(s[3]); \
    ADD32_DPP(s[2], s[3]); s[1] ^= s[2]; ROTR64_63_DPP(s[1]);

#define shflldpp(state) \
	asm(" shfl.sync.idx.b32  %0, %0, %3, 0x1C1F, 0xffffffff;\n\t" \
	    " shfl.sync.idx.b32  %1, %1, %4, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %2, %2, %5, 0x1C1F, 0xffffffff;" \
		: "+r"(state[1]), "+r"(state[2]), "+r"(state[3]) : "r"(LOCAL_LINEAR + 1), "r"(LOCAL_LINEAR + 2), "r"(LOCAL_LINEAR + 3));

#define shflrdpp(state) \
	asm(" shfl.sync.idx.b32  %0, %0, %3, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %1, %1, %4, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %2, %2, %5, 0x1C1F, 0xffffffff;" \
		: "+r"(state[1]), "+r"(state[2]), "+r"(state[3]) : "r"(LOCAL_LINEAR + 3), "r"(LOCAL_LINEAR + 2), "r"(LOCAL_LINEAR + 1));

// pad counts 4 entries each hash team of 4
#define round_lyra_4way_sw(state)   \
	cipher_G_macro(state); \
	shflldpp(state); \
	cipher_G_macro(state);\
	shflrdpp(state);

#define xorrot_one_dpp(sII, state) \
	s0 = state[0]; \
	s1 = state[1]; \
	s2 = state[2]; \
	asm(" shfl.sync.idx.b32  %0, %0, %3, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %1, %1, %3, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %2, %2, %3, 0x1C1F, 0xffffffff;" \
		: "+r"(s0), "+r"(s1), "+r"(s2) : "r"(LOCAL_LINEAR + 3)); \
	if ((threadIdx.x & 3) == 1) sII[0] ^= (s0); \
	if ((threadIdx.x & 3) == 1) sII[1] ^= (s1); \
	if ((threadIdx.x & 3) == 1) sII[2] ^= (s2); \
	if ((threadIdx.x & 3) == 2) sII[0] ^= (s0); \
	if ((threadIdx.x & 3) == 2) sII[1] ^= (s1); \
	if ((threadIdx.x & 3) == 2) sII[2] ^= (s2); \
	if ((threadIdx.x & 3) == 3) sII[0] ^= (s0); \
	if ((threadIdx.x & 3) == 3) sII[1] ^= (s1); \
	if ((threadIdx.x & 3) == 3) sII[2] ^= (s2); \
	if ((threadIdx.x & 3) == 0) sII[0] ^= (s2); \
	if ((threadIdx.x & 3) == 0) sII[1] ^= (s0); \
	if ((threadIdx.x & 3) == 0) sII[2] ^= (s1); \

#define broadcast_zero(s) \
    p0 = (s[0] & 7); \
	asm(" shfl.sync.idx.b32  %0, %0, 0x0, 0x181F, 0xffffffff;" \
		: "+r"(p0) :); \
	if ((threadIdx.x & 2) == 0) modify = p0; \
	if ((threadIdx.x & 2) == 2) modify = p0;

#define write_state(notepad, state, row, col) \
  notepad[24 * row + col * 3] = state[0]; \
  notepad[24 * row + col * 3 + 1] = state[1]; \
  notepad[24 * row + col * 3 + 2] = state[2];

#define state_xor_modify(modify, row, col, mindex, state, notepad) \
  if (modify == row) state[0] ^= notepad[24 * row + col * 3]; \
  if (modify == row) state[1] ^= notepad[24 * row + col * 3 + 1]; \
  if (modify == row) state[2] ^= notepad[24 * row + col * 3 + 2];

#define state_xor(state, bigMat, mindex, row, col) \
  si[0] = bigMat[24 * row + col * 3]; state[0] ^= bigMat[24 * row + col * 3]; \
  si[1] = bigMat[24 * row + col * 3 + 1]; state[1] ^= bigMat[24 * row + col * 3 + 1]; \
  si[2] = bigMat[24 * row + col * 3 + 2]; state[2] ^= bigMat[24 * row + col * 3 + 2];

#define xor_state(state, bigMat, mindex, row, col) \
  si[0] ^= state[0]; bigMat[24 * row + col * 3] = si[0]; \
  si[1] ^= state[1]; bigMat[24 * row + col * 3 + 1] = si[1]; \
  si[2] ^= state[2]; bigMat[24 * row + col * 3 + 2] = si[2];

#define state_xor_plus(state, bigMat, mindex, matin, colin, matrw, colrw) \
   si[0] = bigMat[24 * matin + colin * 3]; sII[0] = bigMat[24 * matrw + colrw * 3]; ss = si[0]; ADD32_DPP(ss, sII[0]); state[0] ^= ss; \
   si[1] = bigMat[24 * matin + colin * 3 + 1]; sII[1] = bigMat[24 * matrw + colrw * 3 + 1]; ss = si[1]; ADD32_DPP(ss, sII[1]); state[1] ^= ss; \
   si[2] = bigMat[24 * matin + colin * 3 + 2]; sII[2] = bigMat[24 * matrw + colrw * 3 + 2]; ss = si[2]; ADD32_DPP(ss, sII[2]); state[2] ^= ss;

#define make_hyper_one_macro(state, bigMat) do { \
    { \
		state_xor(state, bigMat, mindex, 0, 0); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 7); \
		state_xor(state, bigMat, mindex, 0, 1); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 6); \
		state_xor(state, bigMat, mindex, 0, 2); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 5); \
		state_xor(state, bigMat, mindex, 0, 3); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 4); \
		state_xor(state, bigMat, mindex, 0, 4); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 3); \
		state_xor(state, bigMat, mindex, 0, 5); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 2); \
		state_xor(state, bigMat, mindex, 0, 6); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 1); \
		state_xor(state, bigMat, mindex, 0, 7); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, 1, 0); \
	} \
} while (0);

#define make_next_hyper_macro(matin, matrw, matout, state, bigMat) do { \
	{ \
		state_xor_plus(state, bigMat, mindex, matin, 0, matrw, 0); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 7); \
		xorrot_one_dpp(sII, state); \
		write_state(bigMat, sII, matrw, 0); \
		state_xor_plus(state, bigMat, mindex, matin, 1, matrw, 1); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 6); \
		xorrot_one_dpp(sII, state); \
        write_state(bigMat, sII, matrw, 1); \
		state_xor_plus(state, bigMat, mindex, matin, 2, matrw, 2); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 5); \
		xorrot_one_dpp(sII, state); \
        write_state(bigMat, sII, matrw, 2); \
		state_xor_plus(state, bigMat, mindex, matin, 3, matrw, 3); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 4); \
		xorrot_one_dpp(sII, state); \
        write_state(bigMat, sII, matrw, 3); \
		state_xor_plus(state, bigMat, mindex, matin, 4, matrw, 4); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 3); \
		xorrot_one_dpp(sII, state); \
        write_state(bigMat, sII, matrw, 4); \
		state_xor_plus(state, bigMat, mindex, matin, 5, matrw, 5); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 2); \
		xorrot_one_dpp(sII, state); \
        write_state(bigMat, sII, matrw, 5); \
		state_xor_plus(state, bigMat, mindex, matin, 6, matrw, 6); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 1); \
		xorrot_one_dpp(sII, state); \
        write_state(bigMat, sII, matrw, 6); \
		state_xor_plus(state, bigMat, mindex, matin, 7, matrw, 7); \
		round_lyra_4way_sw(state); \
		xor_state(state, bigMat, mindex, matout, 0); \
		xorrot_one_dpp(sII, state); \
        write_state(bigMat, sII, matrw, 7); \
	} \
} while (0);

#define real_matrw_read(sII, bigMat, matrw, off) \
		if (matrw == 0) sII[0] = bigMat[24 * 0 + off * 3]; \
		if (matrw == 0) sII[1] = bigMat[24 * 0 + off * 3 + 1]; \
		if (matrw == 0) sII[2] = bigMat[24 * 0 + off * 3 + 2]; \
		if (matrw == 1) sII[0] = bigMat[24 * 1 + off * 3]; \
		if (matrw == 1) sII[1] = bigMat[24 * 1 + off * 3 + 1]; \
		if (matrw == 1) sII[2] = bigMat[24 * 1 + off * 3 + 2]; \
		if (matrw == 2) sII[0] = bigMat[24 * 2 + off * 3]; \
		if (matrw == 2) sII[1] = bigMat[24 * 2 + off * 3 + 1]; \
		if (matrw == 2) sII[2] = bigMat[24 * 2 + off * 3 + 2]; \
		if (matrw == 3) sII[0] = bigMat[24 * 3 + off * 3]; \
		if (matrw == 3) sII[1] = bigMat[24 * 3 + off * 3 + 1]; \
		if (matrw == 3) sII[2] = bigMat[24 * 3 + off * 3 + 2]; \
		if (matrw == 4) sII[0] = bigMat[24 * 4 + off * 3]; \
		if (matrw == 4) sII[1] = bigMat[24 * 4 + off * 3 + 1]; \
		if (matrw == 4) sII[2] = bigMat[24 * 4 + off * 3 + 2]; \
		if (matrw == 5) sII[0] = bigMat[24 * 5 + off * 3]; \
		if (matrw == 5) sII[1] = bigMat[24 * 5 + off * 3 + 1]; \
		if (matrw == 5) sII[2] = bigMat[24 * 5 + off * 3 + 2]; \
		if (matrw == 6) sII[0] = bigMat[24 * 6 + off * 3]; \
		if (matrw == 6) sII[1] = bigMat[24 * 6 + off * 3 + 1]; \
		if (matrw == 6) sII[2] = bigMat[24 * 6 + off * 3 + 2]; \
		if (matrw == 7) sII[0] = bigMat[24 * 7 + off * 3]; \
		if (matrw == 7) sII[1] = bigMat[24 * 7 + off * 3 + 1]; \
		if (matrw == 7) sII[2] = bigMat[24 * 7 + off * 3 + 2];

#define real_matrw_write(sII, bigMat, matrw, off) \
		if (matrw == 0) bigMat[24 * 0 + off * 3] = sII[0]; \
		if (matrw == 0) bigMat[24 * 0 + off * 3 + 1] = sII[1]; \
		if (matrw == 0) bigMat[24 * 0 + off * 3 + 2] = sII[2]; \
		if (matrw == 1) bigMat[24 * 1 + off * 3] = sII[0]; \
		if (matrw == 1) bigMat[24 * 1 + off * 3 + 1] = sII[1]; \
		if (matrw == 1) bigMat[24 * 1 + off * 3 + 2] = sII[2]; \
		if (matrw == 2) bigMat[24 * 2 + off * 3] = sII[0]; \
		if (matrw == 2) bigMat[24 * 2 + off * 3 + 1] = sII[1]; \
		if (matrw == 2) bigMat[24 * 2 + off * 3 + 2] = sII[2]; \
		if (matrw == 3) bigMat[24 * 3 + off * 3] = sII[0]; \
		if (matrw == 3) bigMat[24 * 3 + off * 3 + 1] = sII[1]; \
		if (matrw == 3) bigMat[24 * 3 + off * 3 + 2] = sII[2]; \
		if (matrw == 4) bigMat[24 * 4 + off * 3] = sII[0]; \
		if (matrw == 4) bigMat[24 * 4 + off * 3 + 1] = sII[1]; \
		if (matrw == 4) bigMat[24 * 4 + off * 3 + 2] = sII[2]; \
		if (matrw == 5) bigMat[24 * 5 + off * 3] = sII[0]; \
		if (matrw == 5) bigMat[24 * 5 + off * 3 + 1] = sII[1]; \
		if (matrw == 5) bigMat[24 * 5 + off * 3 + 2] = sII[2]; \
		if (matrw == 6) bigMat[24 * 6 + off * 3] = sII[0]; \
		if (matrw == 6) bigMat[24 * 6 + off * 3 + 1] = sII[1]; \
		if (matrw == 6) bigMat[24 * 6 + off * 3 + 2] = sII[2]; \
		if (matrw == 7) bigMat[24 * 7 + off * 3] = sII[0]; \
		if (matrw == 7) bigMat[24 * 7 + off * 3 + 1] = sII[1]; \
		if (matrw == 7) bigMat[24 * 7 + off * 3 + 2] = sII[2];

#define state_xor_plus_modify(state, bigMat, mindex, matin, colin, matrw, colrw) \
   si[0] = bigMat[24 * matin + colin * 3]; \
   si[1] = bigMat[24 * matin + colin * 3 + 1]; \
   si[2] = bigMat[24 * matin + colin * 3 + 2]; \
   real_matrw_read(sII, bigMat, matrw, colrw); \
   ss = si[0]; ADD32_DPP(ss, sII[0]); state[0] ^= ss; \
   ss = si[1]; ADD32_DPP(ss, sII[1]); state[1] ^= ss; \
   ss = si[2]; ADD32_DPP(ss, sII[2]); state[2] ^= ss;

#define xor_state_modify(state, bigMat, mindex, row, col) \
  bigMat[24 * row + col * 3] ^= state[0]; \
  bigMat[24 * row + col * 3 + 1] ^= state[1]; \
  bigMat[24 * row + col * 3 + 2] ^= state[2];

#define hyper_xor_dpp_macro( matin, matrw, matout, state, bigMat) do { \
    { \
		state_xor_plus_modify(state, bigMat, mindex, matin, 0, matrw, 0); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 0); xor_state_modify(state, bigMat, mindex, matout, 0); \
		state_xor_plus_modify(state, bigMat, mindex, matin, 1, matrw, 1); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 1); xor_state_modify(state, bigMat, mindex, matout, 1); \
		state_xor_plus_modify(state, bigMat, mindex, matin, 2, matrw, 2); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 2); xor_state_modify(state, bigMat, mindex, matout, 2); \
		state_xor_plus_modify(state, bigMat, mindex, matin, 3, matrw, 3); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 3); xor_state_modify(state, bigMat, mindex, matout, 3); \
		state_xor_plus_modify(state, bigMat, mindex, matin, 4, matrw, 4); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 4); xor_state_modify(state, bigMat, mindex, matout, 4); \
		state_xor_plus_modify(state, bigMat, mindex, matin, 5, matrw, 5); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 5); xor_state_modify(state, bigMat, mindex, matout, 5); \
		state_xor_plus_modify(state, bigMat, mindex, matin, 6, matrw, 6); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 6); xor_state_modify(state, bigMat, mindex, matout, 6); \
		state_xor_plus_modify(state, bigMat, mindex, matin, 7, matrw, 7); \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 7); xor_state_modify(state, bigMat, mindex, matout, 7); \
	} \
} while (0);

__global__
__launch_bounds__(32, 1)
void lyra2_gpu_hash_fancyIX_32_2(uint32_t threads, uint32_t startNounce)
{
	const uint32_t thread = blockDim.z * blockIdx.z + threadIdx.z;

	if (thread < threads)
	{

		const unsigned int LOCAL_LINEAR = threadIdx.x & 3;
		const unsigned int player = threadIdx.y & 1;
		const unsigned int warp_local = LOCAL_LINEAR + 4 * player;

		unsigned int notepad[192];

		unsigned int zero = threadIdx.x ;
		unsigned int state[4];
		unsigned int si[3];
		unsigned int sII[3];
		unsigned int s0;
		  unsigned int s1;
		  unsigned int s2;
		unsigned int ss;

		if (LOCAL_LINEAR == 0) state[0] = __ldg(&(((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 0) + player]));
		if (LOCAL_LINEAR == 0) state[1] = __ldg(&(((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 0) + player]));
		if (LOCAL_LINEAR == 0) state[2] = __ldg(&(((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 0) + player]));
		if (LOCAL_LINEAR == 0) state[3] = __ldg(&(((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 0) + player]));
		if (LOCAL_LINEAR == 1) state[0] = __ldg(&(((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 1) + player]));
		if (LOCAL_LINEAR == 1) state[1] = __ldg(&(((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 1) + player]));
		if (LOCAL_LINEAR == 1) state[2] = __ldg(&(((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 1) + player]));
		if (LOCAL_LINEAR == 1) state[3] = __ldg(&(((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 1) + player]));
		if (LOCAL_LINEAR == 2) state[0] = __ldg(&(((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 2) + player]));
		if (LOCAL_LINEAR == 2) state[1] = __ldg(&(((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 2) + player]));
		if (LOCAL_LINEAR == 2) state[2] = __ldg(&(((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 2) + player]));
		if (LOCAL_LINEAR == 2) state[3] = __ldg(&(((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 2) + player]));
		if (LOCAL_LINEAR == 3) state[0] = __ldg(&(((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 3) + player]));
		if (LOCAL_LINEAR == 3) state[1] = __ldg(&(((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 3) + player]));
		if (LOCAL_LINEAR == 3) state[2] = __ldg(&(((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 3) + player]));
		if (LOCAL_LINEAR == 3) state[3] = __ldg(&(((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 3) + player]));

		write_state(notepad, state, 0, 7);
		round_lyra_4way_sw(state);
		
		write_state(notepad, state, 0, 6);
		round_lyra_4way_sw(state);
		write_state(notepad, state, 0, 5);
		round_lyra_4way_sw(state);
		write_state(notepad, state, 0, 4);
		round_lyra_4way_sw(state);
		write_state(notepad, state, 0, 3);
		round_lyra_4way_sw(state);
		write_state(notepad, state, 0, 2);
		round_lyra_4way_sw(state);
		write_state(notepad, state, 0, 1);
		round_lyra_4way_sw(state);
		write_state(notepad, state, 0, 0);
		round_lyra_4way_sw(state);
		
		make_hyper_one_macro(state, notepad);
		
		make_next_hyper_macro(1, 0, 2, state, notepad);
		
		make_next_hyper_macro(2, 1, 3, state, notepad);
		make_next_hyper_macro(3, 0, 4, state, notepad);
		make_next_hyper_macro(4, 3, 5, state, notepad);
		make_next_hyper_macro(5, 2, 6, state, notepad);
		make_next_hyper_macro(6, 1, 7, state, notepad);
	  
		unsigned int modify = 0;
		unsigned int p0;
	  
		broadcast_zero(state);
		hyper_xor_dpp_macro(7, modify, 0, state, notepad);
		
		broadcast_zero(state);
		hyper_xor_dpp_macro(0, modify, 3, state, notepad);
		
		broadcast_zero(state);
		hyper_xor_dpp_macro(3, modify, 6, state, notepad);
		
		broadcast_zero(state);
		hyper_xor_dpp_macro(6, modify, 1, state, notepad);
		broadcast_zero(state);
		hyper_xor_dpp_macro(1, modify, 4, state, notepad);
		broadcast_zero(state);
		hyper_xor_dpp_macro(4, modify, 7, state, notepad);
		broadcast_zero(state);
		hyper_xor_dpp_macro(7, modify, 2, state, notepad);
		broadcast_zero(state);
		hyper_xor_dpp_macro(2, modify, 5, state, notepad);
	  
		state_xor_modify(modify, 0, 0, mindex, state, notepad);
		state_xor_modify(modify, 1, 0, mindex, state, notepad);
		state_xor_modify(modify, 2, 0, mindex, state, notepad);
		state_xor_modify(modify, 3, 0, mindex, state, notepad);
		state_xor_modify(modify, 4, 0, mindex, state, notepad);
		state_xor_modify(modify, 5, 0, mindex, state, notepad);
		state_xor_modify(modify, 6, 0, mindex, state, notepad);
		state_xor_modify(modify, 7, 0, mindex, state, notepad);
	  
		zero = 1;

		if (LOCAL_LINEAR == 0) ((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 0) + player] = state[0];
		if (LOCAL_LINEAR == 0) ((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 0) + player] = state[1];
		if (LOCAL_LINEAR == 0) ((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 0) + player] = state[2];
		if (LOCAL_LINEAR == 0) ((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 0) + player] = state[3];
		if (LOCAL_LINEAR == 1) ((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 1) + player] = state[0];
		if (LOCAL_LINEAR == 1) ((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 1) + player] = state[1];
		if (LOCAL_LINEAR == 1) ((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 1) + player] = state[2];
		if (LOCAL_LINEAR == 1) ((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 1) + player] = state[3];
		if (LOCAL_LINEAR == 2) ((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 2) + player] = state[0];
		if (LOCAL_LINEAR == 2) ((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 2) + player] = state[1];
		if (LOCAL_LINEAR == 2) ((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 2) + player] = state[2];
		if (LOCAL_LINEAR == 2) ((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 2) + player] = state[3];
		if (LOCAL_LINEAR == 3) ((unsigned int *)DMatrix)[2 *((0 * threads + thread) * blockDim.x + 3) + player] = state[0];
		if (LOCAL_LINEAR == 3) ((unsigned int *)DMatrix)[2 *((1 * threads + thread) * blockDim.x + 3) + player] = state[1];
		if (LOCAL_LINEAR == 3) ((unsigned int *)DMatrix)[2 *((2 * threads + thread) * blockDim.x + 3) + player] = state[2];
		if (LOCAL_LINEAR == 3) ((unsigned int *)DMatrix)[2 *((3 * threads + thread) * blockDim.x + 3) + player] = state[3];
	}
}

#else
#define ADD32_DPP(a, b)
#define SWAP32_DPP(s)
#define ROTR64_24_DPP(s) 
#define ROTR64_16_DPP(s) 
#define ROTR64_63_DPP(s)
#define cipher_G_macro(s) 
#define shflldpp(state)
#define shflrdpp(state)
#define round_lyra_4way_sw(state) 
#define xorrot_one_dpp(sII, state)
#define broadcast_zero(s) 
#define write_state(notepad, state, row, col)
#define state_xor_modify(modify, row, col, mindex, state, notepad) 
#define state_xor(state, bigMat, mindex, row, col)
#define xor_state(state, bigMat, mindex, row, col)
#define state_xor_plus(state, bigMat, mindex, matin, colin, matrw, colrw) 
#define make_hyper_one_macro(state, bigMat) 
#define make_next_hyper_macro(matin, matrw, matout, state, bigMat) 
#define real_matrw_read(sII, bigMat, matrw, off)
#define real_matrw_write(sII, bigMat, matrw, off) 
#define state_xor_plus_modify(state, bigMat, mindex, matin, colin, matrw, colrw)
#define xor_state_modify(state, bigMat, mindex, row, col)
#define hyper_xor_dpp_macro( matin, matrw, matout, state, bigMat)
__global__ void lyra2_gpu_hash_fancyIX_32_2(uint32_t threads, uint32_t startNounce) {}

#endif
// =============================================================

__host__
void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, bool gtx750ti)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb = 32;

	if (cuda_arch[dev_id] >= 500) tpb = 32;
	else if (cuda_arch[dev_id] >= 200) tpb = TPB20;

	dim3 grid1(1, 1, (threads * 8 + tpb - 1) / tpb);
	dim3 block1(4, 2, tpb >> 3);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	dim3 grid3((threads + tpb - 1) / tpb);
	dim3 block3(tpb);

	if (cuda_arch[dev_id] >= 520)
	{
		lyra2_gpu_hash_32_1 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

		lyra2_gpu_hash_fancyIX_32_2 <<< grid1, block1 >>> (threads, startNounce);

		lyra2_gpu_hash_32_3 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);
	}
	else if (cuda_arch[dev_id] >= 500)
	{
		lyra2_gpu_hash_32_1_sm5 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

		lyra2_gpu_hash_fancyIX_32_2 <<< grid1, block1 >>> (threads, startNounce);

		lyra2_gpu_hash_32_3_sm5 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);
	}
	else
		lyra2_gpu_hash_32_sm2 <<< grid3, block3 >>> (threads, startNounce, d_hash);
}
