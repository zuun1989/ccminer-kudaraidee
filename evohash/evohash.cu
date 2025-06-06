extern "C" {
#include "sph/sph_groestl.h"
#include "sph/sph_keccak.h"
#include "sph/sph_jh.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_luffa.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_shabal.h"
#include "lyra2/Lyra2.h"
}

#include <miner.h>
#include <cuda_helper.h>
#include "x16/cuda_x16.h"

static uint32_t* d_hash[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];
static uint64_t* d_hash_256[MAX_GPUS];

extern void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2_cuda_hash_64(int thr_id, const uint32_t threads, uint64_t* d_hash_256, uint32_t* d_hash, bool gtx750ti);


extern "C" void evohash(void *state, const void *input)
{
	sph_groestl512_context   ctx_groestl;
	sph_keccak512_context    ctx_keccak;
	sph_cubehash512_context  ctx_cubehash;	
	sph_luffa512_context     ctx_luffa;
	sph_echo512_context      ctx_echo;		
	sph_simd512_context      ctx_simd;	
	sph_shavite512_context   ctx_shavite;
	sph_hamsi512_context     ctx_hamsi;
	sph_fugue512_context     ctx_fugue;
	sph_whirlpool_context    ctx_whirlpool; 
	sph_skein512_context     ctx_skein;
	sph_shabal512_context    ctx_shabal;	
	sph_bmw512_context       ctx_bmw;	
	sph_jh512_context        ctx_jh;
	
	unsigned char hash[128] = { 0 };
	unsigned char hashA[64] = { 0 };
	unsigned char hashB[64] = { 0 };

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, input, 80);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, hashA, 64);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, (const void*) hash, 64);
	sph_hamsi512_close(&ctx_hamsi, hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	
	
	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*) hashA, 64);
	sph_fugue512_close(&ctx_fugue, hash);	
	
	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	
	
	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*) hashA, 64);
	sph_echo512_close(&ctx_echo, hash);	

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	
	
	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*) hashA, 64);
	sph_shavite512_close(&ctx_shavite, hash);
	
	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, (const void*) hash, 64);
	sph_luffa512_close (&ctx_luffa, hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	
	
	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*) hashA, 64);
	sph_shavite512_close(&ctx_shavite, hash);
	
	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, (const void*) hash, 64);
	sph_luffa512_close (&ctx_luffa, hashB);
	
	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	
	
	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool (&ctx_whirlpool, (const void*) hashA, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, (const void*) hash, 64);
	sph_shabal512_close(&ctx_shabal, hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	
	
	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*) hashA, 64);
	sph_jh512_close(&ctx_jh, hash);		

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, hash, 64);
	sph_keccak512_close(&ctx_keccak, hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);	
	
	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, (const void*)hashA, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);	
			      
	for (int i=0; i<32; i++)
		hash[i] ^= hash[i+32];

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static __thread uint32_t throughput = 0;
static __thread bool gtx750ti = false;
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_evohash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[32];

	if (opt_benchmark)
		ptarget[7] = 0xff;
		
	if (!init[thr_id])
	{
		int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 19 : 18;
		if (device_sm[device_map[thr_id]] == 500) intensity = 17;
		throughput = cuda_default_throughput(thr_id, 1U << intensity);
		if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
		if (init[thr_id]) throughput = max(throughput & 0xffffff80, 128); // for shared mem
		
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);

		if (strstr(props.name, "750 Ti")) gtx750ti = true;
		else gtx750ti = false;

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			x11_echo512_cpu_init(thr_id, throughput);
		
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x16_echo512_cuda_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);

		if (device_sm[dev_id] >= 500)
		{
			size_t matrix_sz = device_sm[dev_id] > 500 ? sizeof(uint64_t) * 16 : sizeof(uint64_t) * 8 * 8 * 3 * 4;
			CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
			CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash_256[thr_id], (size_t)32 * throughput), -1);
			lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);
		}

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	cubehash512_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;
		cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id], gtx750ti); order++;
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id], gtx750ti); order++;
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id], gtx750ti); order++;
		if (use_compat_kernels[thr_id])
		{
			x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		}else {
			x16_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		}
		x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id], gtx750ti); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id], gtx750ti); order++;
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id], gtx750ti); order++;
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id], gtx750ti); order++;
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			evohash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					evohash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_evohash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);
	cudaFree(d_hash_256[thr_id]);
	
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
