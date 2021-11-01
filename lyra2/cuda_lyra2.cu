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
/// lyra2 algo  ///////////////////////////////////////////////////////////
#define HASH_SIZE (256 / 8) // size in bytes of an hash in/out
#define LOCAL_LINEAR (threadIdx.x & 3)
#define REG_ROW_COUNT (1) // ideally all happen at the same clock
#define STATE_BLOCK_COUNT (1 * REG_ROW_COUNT)  // very close instructions
#define LYRA_ROUNDS 8
#define HYPERMATRIX_COUNT (LYRA_ROUNDS * STATE_BLOCK_COUNT)
#define ROTR64F(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#define SWAPL(x)  (((x) << 32) | ((x) >> 32))
// Usually just #define G(a,b,c,d)...; I have no time to read the Lyra paper
// but that looks like some kind of block cipher I guess.
#define cipher_G_macro(s) \
	s[0] += s[1]; s[3] ^= s[0]; s[3] = SWAPL(s[3]); \
	s[2] += s[3]; s[1] ^= s[2]; s[1] = ROTR64F(s[1], 24); \
	s[0] += s[1]; s[3] ^= s[0]; s[3] = ROTR64F(s[3], 16); \
	s[2] += s[3]; s[1] ^= s[2]; s[1] = ROTR64F(s[1], 63);

#define pull_state(state) \
	s0 = (*reinterpret_cast<uint2 *>(&(cstate))); \
	s1 = (*reinterpret_cast<uint2 *>(&(cstate))); \
	s2 = (*reinterpret_cast<uint2 *>(&(cstate))); \
	s3 = (*reinterpret_cast<uint2 *>(&(cstate))); \
	asm(" shfl.sync.idx.b32  %0, %0, %8, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %1, %1, %8, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %2, %2, %9, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %3, %3, %9, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %4, %4, %10, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %5, %5, %10, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %6, %6, %11, 0x1C1F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %7, %7, %11, 0x1C1F, 0xffffffff;" \
		: "+r"(s0.x), "+r"(s0.y), "+r"(s1.x), "+r"(s1.y), "+r"(s2.x), "+r"(s2.y), "+r"(s3.x), "+r"(s3.y) \
		: "r"(0), "r"(1), "r"(2), "r"(3)); \
	state[0] = (*reinterpret_cast<ulong *>(&(s0))); \
	state[1] = (*reinterpret_cast<ulong *>(&(s1))); \
	state[2] = (*reinterpret_cast<ulong *>(&(s2))); \
	state[3] = (*reinterpret_cast<ulong *>(&(s3)));

#define shflldpp(state) \
	s1 = (*reinterpret_cast<uint2 *>(&(state[1]))); \
	s2 = (*reinterpret_cast<uint2 *>(&(state[2]))); \
	s3 = (*reinterpret_cast<uint2 *>(&(state[3]))); \
	asm(" shfl.sync.idx.b32  %0, %0, %6, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %1, %1, %6, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %2, %2, %7, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %3, %3, %7, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %4, %4, %8, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %5, %5, %8, 0x101F, 0xffffffff;" \
		: "+r"(s1.x), "+r"(s1.y), "+r"(s2.x), "+r"(s2.y), "+r"(s3.x), "+r"(s3.y) \
		: "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 4), "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 8), "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 12)); \
	state[1] = (*reinterpret_cast<ulong *>(&(s1))); \
	state[2] = (*reinterpret_cast<ulong *>(&(s2))); \
	state[3] = (*reinterpret_cast<ulong *>(&(s3)));

#define shflrdpp(state)  \
	s1 = (*reinterpret_cast<uint2 *>(&(state[1]))); \
	s2 = (*reinterpret_cast<uint2 *>(&(state[2]))); \
	s3 = (*reinterpret_cast<uint2 *>(&(state[3]))); \
	asm(" shfl.sync.idx.b32  %0, %0, %6, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %1, %1, %6, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %2, %2, %7, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %3, %3, %7, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %4, %4, %8, 0x101F, 0xffffffff;\n\t" \
		" shfl.sync.idx.b32  %5, %5, %8, 0x101F, 0xffffffff;" \
		: "+r"(s1.x), "+r"(s1.y), "+r"(s2.x), "+r"(s2.y), "+r"(s3.x), "+r"(s3.y) \
		: "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 12), "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 8), "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 4)); \
	state[1] = (*reinterpret_cast<ulong *>(&(s1))); \
	state[2] = (*reinterpret_cast<ulong *>(&(s2))); \
	state[3] = (*reinterpret_cast<ulong *>(&(s3)));

// pad counts 4 entries each hash team of 4
#define round_lyra_4way_sw(state)   \
	pull_state(state); \
	cipher_G_macro(state); \
	shflldpp(state); \
	cipher_G_macro(state);\
	shflrdpp(state); \
	if (LOCAL_LINEAR == 0) cstate = state[0]; \
	if (LOCAL_LINEAR == 1) cstate = state[1]; \
	if (LOCAL_LINEAR == 2) cstate = state[2]; \
	if (LOCAL_LINEAR == 3) cstate = state[3];

#define xorrot_one_dpp(sII, state) \
	s0 = (*reinterpret_cast<uint2 *>(&(state[0]))); \
	s1 = (*reinterpret_cast<uint2 *>(&(state[1]))); \
	s2 = (*reinterpret_cast<uint2 *>(&(state[2]))); \
	asm(" shfl.sync.idx.b32  %0, %0, %6, 0x101F, 0xffffffff;\n\t" \
	" shfl.sync.idx.b32  %1, %1, %6, 0x101F, 0xffffffff;\n\t" \
	" shfl.sync.idx.b32  %2, %2, %6, 0x101F, 0xffffffff;\n\t" \
	" shfl.sync.idx.b32  %3, %3, %6, 0x101F, 0xffffffff;\n\t" \
	" shfl.sync.idx.b32  %4, %4, %6, 0x101F, 0xffffffff;\n\t" \
	" shfl.sync.idx.b32  %5, %5, %6, 0x101F, 0xffffffff;" \
	: "+r"(s0.x), "+r"(s0.y), "+r"(s1.x), "+r"(s1.y), "+r"(s2.x), "+r"(s2.y) \
	: "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 12)); \
	if ((threadIdx.y & 3) == 1 || (threadIdx.y & 3) == 2 || (threadIdx.y & 3) == 3) if (LOCAL_LINEAR == 0) sII ^= (*reinterpret_cast<ulong *>(&(s0))); \
	if ((threadIdx.y & 3) == 1 || (threadIdx.y & 3) == 2 || (threadIdx.y & 3) == 3) if (LOCAL_LINEAR == 1) sII ^= (*reinterpret_cast<ulong *>(&(s1))); \
	if ((threadIdx.y & 3) == 1 || (threadIdx.y & 3) == 2 || (threadIdx.y & 3) == 3) if (LOCAL_LINEAR == 2) sII ^= (*reinterpret_cast<ulong *>(&(s2))); \
	if ((threadIdx.y & 3) == 0 ) if (LOCAL_LINEAR == 0) sII ^= (*reinterpret_cast<ulong *>(&(s2))); \
	if ((threadIdx.y & 3) == 0 ) if (LOCAL_LINEAR == 1) sII ^= (*reinterpret_cast<ulong *>(&(s0))); \
	if ((threadIdx.y & 3) == 0 ) if (LOCAL_LINEAR == 2) sII ^= (*reinterpret_cast<ulong *>(&(s1))); \

#define make_hyper_one_macro(state, bigMat) do { \
	{ \
		si = bigMat[0]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[15] = si; \
		si = bigMat[1]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[14] = si; \
		si = bigMat[2]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[13] = si; \
		si = bigMat[3]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[12] = si; \
		si = bigMat[4]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[11] = si; \
		si = bigMat[5]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[10] = si; \
		si = bigMat[6]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[9] = si; \
		si = bigMat[7]; if (LOCAL_LINEAR != 3) cstate ^= si; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8] = si; \
	} \
} while (0);

#define make_next_hyper_macro(matin, matrw, matout, state, bigMat) do { \
	{ \
		si = bigMat[8 * matin]; sII = bigMat[8 * matrw]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout + 7] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw] = sII; \
		si = bigMat[8 * matin + 1]; sII = bigMat[8 * matrw + 1]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout + 6] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw + 1] = sII; \
		si = bigMat[8 * matin + 2]; sII = bigMat[8 * matrw + 2]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout + 5] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw + 2] = sII; \
		si = bigMat[8 * matin + 3]; sII = bigMat[8 * matrw + 3]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout + 4] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw + 3] = sII; \
		si = bigMat[8 * matin + 4]; sII = bigMat[8 * matrw + 4]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout + 3] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw + 4] = sII; \
		si = bigMat[8 * matin + 5]; sII = bigMat[8 * matrw + 5]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout + 2] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw + 5] = sII; \
		si = bigMat[8 * matin + 6]; sII = bigMat[8 * matrw + 6]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout + 1] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw + 6] = sII; \
		si = bigMat[8 * matin + 7]; sII = bigMat[8 * matrw + 7]; if (LOCAL_LINEAR != 3) cstate ^= si + sII; \
		round_lyra_4way_sw(state); \
		if (LOCAL_LINEAR != 3) si ^= cstate; bigMat[8 * matout] = si; \
		xorrot_one_dpp(sII, state); \
		bigMat[8 * matrw + 7] = sII; \
	} \
} while (0);

#define broadcast_zero(s) \
	p0 = (s[0] & 7); \
	p1 = (s[0] & 7); \
	p2 = (s[0] & 7); \
	p3 = (s[0] & 7); \
	asm(" shfl.sync.idx.b32  %0, %0, %3, 0x101F, 0xffffffff;\n\t" \
	" shfl.sync.idx.b32  %1, %1, %4, 0x101F, 0xffffffff;\n\t" \
	" shfl.sync.idx.b32  %2, %2, %5, 0x101F, 0xffffffff;" \
	: "+r"(p1), "+r"(p2), "+r"(p3) \
	: "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 12), "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 8), "r"(LOCAL_LINEAR + (4 * threadIdx.y) + 4)); \
	if ((threadIdx.y & 3) == 1) modify = p1; \
	if ((threadIdx.y & 3) == 2) modify = p2; \
	if ((threadIdx.y & 3) == 3) modify = p3; \
	if ((threadIdx.y & 3) == 0) modify = p0;

#define real_matrw_read(sII, bigMat, matrw, off) \
		if (matrw == 0) sII = bigMat[8 * 0 + off];  \
		if (matrw == 1) sII = bigMat[8 * 1 + off]; \
		if (matrw == 2) sII = bigMat[8 * 2 + off]; \
		if (matrw == 3) sII = bigMat[8 * 3 + off]; \
		if (matrw == 4) sII = bigMat[8 * 4 + off]; \
		if (matrw == 5) sII = bigMat[8 * 5 + off]; \
		if (matrw == 6) sII = bigMat[8 * 6 + off]; \
		if (matrw == 7) sII = bigMat[8 * 7 + off];

#define real_matrw_write(sII, bigMat, matrw, off) \
		if (matrw == 0) bigMat[8 * 0 + off] = sII; \
		if (matrw == 1) bigMat[8 * 1 + off] = sII; \
		if (matrw == 2) bigMat[8 * 2 + off] = sII; \
		if (matrw == 3) bigMat[8 * 3 + off] = sII; \
		if (matrw == 4) bigMat[8 * 4 + off] = sII; \
		if (matrw == 5) bigMat[8 * 5 + off] = sII; \
		if (matrw == 6) bigMat[8 * 6 + off] = sII; \
		if (matrw == 7) bigMat[8 * 7 + off] = sII;

#define hyper_xor_dpp_macro( matin, matrw, matout, state, bigMat) do { \
	{ \
		si = bigMat[8 * matin + 0]; real_matrw_read(sII, bigMat, matrw, 0); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 0); bigMat[8 * matout + 0] ^= cstate; \
		si = bigMat[8 * matin + 1]; real_matrw_read(sII, bigMat, matrw, 1); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 1); bigMat[8 * matout + 1] ^= cstate; \
		si = bigMat[8 * matin + 2]; real_matrw_read(sII, bigMat, matrw, 2); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 2); bigMat[8 * matout + 2] ^= cstate; \
		si = bigMat[8 * matin + 3]; real_matrw_read(sII, bigMat, matrw, 3); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 3); bigMat[8 * matout + 3] ^= cstate; \
		si = bigMat[8 * matin + 4]; real_matrw_read(sII, bigMat, matrw, 4); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 4); bigMat[8 * matout + 4] ^= cstate; \
		si = bigMat[8 * matin + 5]; real_matrw_read(sII, bigMat, matrw, 5); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 5); bigMat[8 * matout + 5] ^= cstate; \
		si = bigMat[8 * matin + 6]; real_matrw_read(sII, bigMat, matrw, 6); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 6); bigMat[8 * matout + 6] ^= cstate; \
		si = bigMat[8 * matin + 7]; real_matrw_read(sII, bigMat, matrw, 7); if (LOCAL_LINEAR != 3) cstate ^= sII + si; \
		round_lyra_4way_sw(state); \
		xorrot_one_dpp(sII, state); \
		real_matrw_write(sII, bigMat, matrw, 7); bigMat[8 * matout + 7] ^= cstate; \
	} \
} while (0);

__global__
__launch_bounds__(32, 1)
void lyra2_gpu_hash_fancyIX_32_2(uint32_t threads, uint32_t startNounce)
{
	const uint32_t thread = 2 * blockIdx.x + threadIdx.z;

	if (thread < threads)
	{
	   ulong notepad[192 / 3];

	   const int player = threadIdx.y;
	 
	   ulong state[4];
	   ulong cstate;
	   ulong si;
	   ulong sII;
	   uint2 s0;
		 uint2 s1;
		 uint2 s2;
		 uint2 s3;
	 
	   //-------------------------------------
	   // Load Lyra state
	   cstate = __ldg(&((ulong *)DMatrix)[(LOCAL_LINEAR * threads + thread) * blockDim.y + threadIdx.y]);
	 


	   notepad[8 - (0 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   notepad[8 - (1 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   notepad[8 - (2 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   notepad[8 - (3 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   notepad[8 - (4 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   notepad[8 - (5 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   notepad[8 - (6 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   notepad[8 - (7 + 1)] = cstate;
	   round_lyra_4way_sw(state);
	   
	   make_hyper_one_macro(state, notepad);
	   
	   make_next_hyper_macro(1, 0, 2, state, notepad);
	   make_next_hyper_macro(2, 1, 3, state, notepad);
	   make_next_hyper_macro(3, 0, 4, state, notepad);
	   make_next_hyper_macro(4, 3, 5, state, notepad);
	   make_next_hyper_macro(5, 2, 6, state, notepad);
	   make_next_hyper_macro(6, 1, 7, state, notepad);
	 
	   uint modify = 0;
	   uint p0;
	   uint p1;
	   uint p2;
	   uint p3;
	 
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
	 
	   if (modify == 0)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 0];
	   if (modify == 1)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 1];
	   if (modify == 2)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 2];
	   if (modify == 3)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 3];
	   if (modify == 4)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 4];
	   if (modify == 5)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 5];
	   if (modify == 6)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 6];
	   if (modify == 7)  if (LOCAL_LINEAR != 3) cstate ^= notepad[HYPERMATRIX_COUNT * 7];
	   //-------------------------------------
	   // save lyra state    
        // save lyra state    
	   // save lyra state    
	   ((ulong *)DMatrix)[(LOCAL_LINEAR * threads + thread) * blockDim.y + threadIdx.y] = cstate;
	}
}

#else
#define HASH_SIZE (256 / 8) // size in bytes of an hash in/out
 #define LOCAL_LINEAR (threadIdx.x & 3)
 #define REG_ROW_COUNT (1) // ideally all happen at the same clock
 #define STATE_BLOCK_COUNT (1 * REG_ROW_COUNT)  // very close instructions
 #define LYRA_ROUNDS 8
 #define HYPERMATRIX_COUNT (LYRA_ROUNDS * STATE_BLOCK_COUNT)
 #define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
 #define SWAPL(x)  (((x) << 32) | ((x) >> 32))
 #define cipher_G_macro(s)
#define cipher_G_macro(s) 
 #define cipher_G_macro(s)
 #define pull_state(state)
 #define shflldpp(state)
 #define shflrdpp(state)
 #define round_lyra_4way_sw(state)
#define round_lyra_4way_sw(state) 
 #define round_lyra_4way_sw(state)
 #define xorrot_one_dpp(sII, state)
 #define make_hyper_one_macro(state, bigMat)
#define make_hyper_one_macro(state, bigMat) 
 #define make_hyper_one_macro(state, bigMat)
 #define make_next_hyper_macro(matin, matrw, matout, state, bigMat)
#define make_next_hyper_macro(matin, matrw, matout, state, bigMat) 
 #define make_next_hyper_macro(matin, matrw, matout, state, bigMat)
 #define broadcast_zero(s)
 #define real_matrw_read(sII, bigMat, matrw, off)
 #define real_matrw_write(sII, bigMat, matrw, off)
#define real_matrw_write(sII, bigMat, matrw, off) 
 #define real_matrw_write(sII, bigMat, matrw, off)
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

	dim3 grid1((threads * 16 + 32 - 1) / 32);
	dim3 block1(4, 4, 2);

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
