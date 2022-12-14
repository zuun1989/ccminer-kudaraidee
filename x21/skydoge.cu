/**
 * SKYDOGE algorithm (X17 + sha256)
 */

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"

#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"

#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x16/cuda_x16.h"

static uint32_t *d_hash[MAX_GPUS];

extern void x17_sha512_cpu_init(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, const int outlen);

extern void sha256_cpu_hash_64(int thr_id, int threads, uint32_t *d_hash);


// Skydoge CPU Hash (Validation)
extern "C" void skydoge_hash(void *output, const void *input)
{
    	sph_blake512_context     ctx_blake;
    	sph_bmw512_context       ctx_bmw;
    	sph_groestl512_context   ctx_groestl;
    	sph_skein512_context     ctx_skein;
    	sph_jh512_context        ctx_jh;
    	sph_keccak512_context    ctx_keccak;
    	sph_luffa512_context     ctx_luffa1;
    	sph_cubehash512_context  ctx_cubehash1;
    	sph_shavite512_context   ctx_shavite1;
    	sph_simd512_context      ctx_simd1;
    	sph_echo512_context      ctx_echo1;
    	sph_hamsi512_context     ctx_hamsi1;
    	sph_fugue512_context     ctx_fugue1;
    	sph_shabal512_context    ctx_shabal1;
    	sph_whirlpool_context    ctx_whirlpool1;
    	sph_sha512_context       ctx_sha512;
    	sph_sha256_context       ctx_sha;
    	sph_haval256_5_context   ctx_haval;

    	uint32_t hash[16];
    	uint32_t hashA[16];

    	sph_blake512_init(&ctx_blake);
    	sph_blake512 (&ctx_blake, input, 80);
    	sph_blake512_close (&ctx_blake, hash);

    	sph_skein512_init(&ctx_skein);
    	sph_skein512 (&ctx_skein, hash, 64);
    	sph_skein512_close (&ctx_skein, hash);

    	sph_bmw512_init(&ctx_bmw);
    	sph_bmw512 (&ctx_bmw, hash, 64);
		sph_bmw512_close(&ctx_bmw, hash);
	
    	sph_groestl512_init(&ctx_groestl);
    	sph_groestl512 (&ctx_groestl, hash, 64);
    	sph_groestl512_close(&ctx_groestl, hash);

    	sph_jh512_init(&ctx_jh);
    	sph_jh512 (&ctx_jh, hash, 64);
    	sph_jh512_close(&ctx_jh, hash);

    	sph_luffa512_init (&ctx_luffa1);
    	sph_luffa512 (&ctx_luffa1, hash, 64);
    	sph_luffa512_close (&ctx_luffa1, hash);

    	sph_keccak512_init(&ctx_keccak);
    	sph_keccak512 (&ctx_keccak, hash, 64);
    	sph_keccak512_close(&ctx_keccak, hash); 

		sph_simd512_init (&ctx_simd1);
    	sph_simd512 (&ctx_simd1, hash, 64);
    	sph_simd512_close(&ctx_simd1, hash);

    	sph_echo512_init (&ctx_echo1);
    	sph_echo512 (&ctx_echo1, hash, 64);
    	sph_echo512_close(&ctx_echo1, hash);

    	sph_cubehash512_init (&ctx_cubehash1);
    	sph_cubehash512 (&ctx_cubehash1, hash, 64);
    	sph_cubehash512_close(&ctx_cubehash1, hash);

    	sph_shavite512_init (&ctx_shavite1);
    	sph_shavite512 (&ctx_shavite1, hash, 64);
    	sph_shavite512_close(&ctx_shavite1, hash);

    	sph_hamsi512_init (&ctx_hamsi1);
    	sph_hamsi512 (&ctx_hamsi1, hash, 64);
    	sph_hamsi512_close(&ctx_hamsi1, hash);

    	sph_fugue512_init (&ctx_fugue1);
    	sph_fugue512 (&ctx_fugue1, hash, 64);
    	sph_fugue512_close(&ctx_fugue1, hash);

    	sph_shabal512_init (&ctx_shabal1);
    	sph_shabal512 (&ctx_shabal1, hash, 64);
    	sph_shabal512_close(&ctx_shabal1, hash);

    	sph_whirlpool_init (&ctx_whirlpool1);
    	sph_whirlpool (&ctx_whirlpool1, hash, 64);
    	sph_whirlpool_close(&ctx_whirlpool1, hash);

    	sph_sha512_init(&ctx_sha512);
    	sph_sha512(&ctx_sha512, hash, 64);
    	sph_sha512_close(&ctx_sha512, hash);

    	sph_simd512_init (&ctx_simd1);
    	sph_simd512 (&ctx_simd1, hash, 64);
    	sph_simd512_close(&ctx_simd1, hash);

    	sph_whirlpool_init (&ctx_whirlpool1);
    	sph_whirlpool (&ctx_whirlpool1, hash, 64);
    	sph_whirlpool_close(&ctx_whirlpool1, hash);

    	sph_sha256_init(&ctx_sha);
    	sph_sha256 (&ctx_sha, hash, 64);
    	sph_sha256_close(&ctx_sha, hashA);

		for (int i=8;i<16;i++)
			hashA[i]=0;

    	sph_haval256_5_init(&ctx_haval);
    	sph_haval256_5(&ctx_haval, hashA, 64);
    	sph_haval256_5_close(&ctx_haval,hash);

		memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_skydoge(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 19); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x003f;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			x11_echo512_cpu_init(thr_id, throughput);
		
		quark_blake512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		qubit_luffa512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x16_whirlpool512_init(thr_id, throughput);
		x17_sha512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x16_whirlpool512_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x17_haval256_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	int warn = 0;

	do {
		int order = 0;

		// Hash with CUDA
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if (use_compat_kernels[thr_id])
			x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		else {
			x16_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		}
		x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		sha256_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], 256); order++;

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			skydoge_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					skydoge_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				// x11+ coins could do some random error, but not on retry
				gpu_increment_reject(thr_id);
				if (!warn) {
					warn++;
					pdata[19] = work->nonces[0] + 1;
					continue;
				} else {
					if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
					warn = 0;
				}
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_skydoge(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id); // to merge with x13_fugue512 ?
	x15_whirlpool_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
