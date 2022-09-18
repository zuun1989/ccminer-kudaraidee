#include "sph/sph_sha2.h"
#include "miner.h"
#include "cuda_helper.h"

extern void rad_cpu_init(int thr_id);
extern void rad_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, const uint64_t *const data, uint32_t *const h_nounce);
extern void rad_midstate(const uint32_t *data, uint32_t *midstate);

void rad_hash(uint32_t *output, const uint32_t *data, uint32_t nonce)
{
	uint32_t header[20];
	for (int i = 0; i < 20; i++) {
		header[i] = swab32(data[i]);
	}
	header[19] = swab32(nonce);
	sph_sha512_context ctx;
	sph_sha512_256_init(&ctx);
	sph_sha512(&ctx, header, 80);
	uint32_t hash1[16];
	sph_sha512_close(&ctx, hash1);

	uint32_t hash2[16];
	sph_sha512_256_init(&ctx);
	sph_sha512(&ctx, hash1, 32);
	sph_sha512_close(&ctx, hash2);

	for (int i = 0; i < 8; i++) {
		output[i] = hash2[i];
	}
}

static bool init[MAX_GPUS] = { 0 };
int scanhash_rad(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 20);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		ptarget[7] = 0x0005;

	if(!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		rad_cpu_init(thr_id);
		init[thr_id] = true;
	}

	do
	{
		rad_cpu_hash(thr_id, throughput, pdata[19], (uint64_t *)pdata, work->nonces);
		if(work->nonces[0] != UINT32_MAX)
		{
			uint32_t vhash64[8]={0};

			rad_hash(vhash64, pdata, work->nonces[0]);
			if (vhash64[7] == 0 && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (work->nonces[1] != 0xffffffff)
				{
					rad_hash(vhash64, pdata, work->nonces[1]);
					if (vhash64[7] == 0 && fulltest(vhash64, ptarget))
					{
						pdata[21] = work->nonces[1];
						res++;
						if (opt_benchmark)
							applog(LOG_INFO, "GPU #%d Found second nounce %08x", device_map[thr_id], work->nonces[1]);
					}
					else
					{
						if (vhash64[7] > 0)
						{
							applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], work->nonces[1]);
						}
					}
				}
				pdata[19] = work->nonces[0];
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", device_map[thr_id], work->nonces[0]);
				return res;
			}
			else
			{
				if (vhash64[7] > 0)
				{
					applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], work->nonces[0]);
				}
			}
		}
		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart);
	
	*hashes_done = pdata[19] - first_nonce ;
	return 0;
}
